# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import math
import pprint
import time
import warnings
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Literal

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean

from litgpt import Tokenizer
from litgpt.args import EvalArgs, KresArgs, LogArgs, TrainArgs
from litgpt.config import name_to_config
from litgpt.constants import _TORCH_EQUAL_2_7, _TORCH_EQUAL_2_8
from litgpt.data import DataModule, TinyLlama
from litgpt.model import GPT, Block, CausalSelfAttention, Config, LLaMAMLP
from litgpt.parser_config import save_hyperparameters
from litgpt.types import LoggerChoice
from litgpt.utils import (
    CycleIterator,
    capture_hparams,
    check_nvlink_connectivity,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    extend_checkpoint_dir,
    find_resume_path,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    num_parameters,
    parse_devices,
    reset_parameters,
    save_config,
)


def setup(
    model_name: str,
    model_config: Config | None = None,
    out_dir: Path = Path("out/pretrain"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
    initial_checkpoint_dir: Path | None = None,
    resume: bool | Literal["auto"] | Path = False,
    data: DataModule | None = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=512,
        micro_batch_size=4,
        max_tokens=int(3e12),  # 3 trillion
        max_norm=1.0,
        min_lr=4e-5,
        lr_warmup_steps=2000,
        tie_embeddings=False,
    ),
    eval: EvalArgs = EvalArgs(interval=1000, max_iters=100),
    log: LogArgs = LogArgs(),
    kres: KresArgs = KresArgs(),
    optimizer: str | dict = "AdamW",
    devices: int | str = "auto",
    num_nodes: int = 1,
    tokenizer_dir: Path | None = None,
    logger_name: LoggerChoice = "tensorboard",
    seed: int = 42,
):
    """Pretrain a model.

    Arguments:
        model_name: The name of the model to pretrain. Choose from names in ``litgpt.config``. Use "list" to list the supported models.
        model_config: A ``litgpt.Config`` object to define the model architecture. Mutually exclusive with
            ``model_config``. Overrides the `model_name` if specified.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Determines a compatible precision setting by default.
        initial_checkpoint_dir: Optional path to a checkpoint directory to initialize the model from.
            Useful for continued pretraining. Mutually exclusive with ``resume``.
        resume: Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
            from the latest checkpoint in ``out_dir``. An error will be raised if no checkpoint is found. Passing
            ``'auto'`` will resume from the latest checkpoint but not error if no checkpoint exists.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.TinyLlama``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.

        devices: How many devices/GPUs to use. Uses all GPUs by default.
        num_nodes: How many nodes the code is being run on.
        tokenizer_dir: Optional path to the tokenizer dir that was used for preprocessing the dataset. Only some data
            module require this.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
    """
    if model_name == "list":
        available_models = "\n".join(sorted(name_to_config))
        print(f"Available values:\n{available_models}")
        quit()

    if initial_checkpoint_dir is not None:
        initial_checkpoint_dir = extend_checkpoint_dir(initial_checkpoint_dir)

    if tokenizer_dir is not None:
        tokenizer_dir = extend_checkpoint_dir(tokenizer_dir)

    if model_config is None:
        # Support both model_name options: meta-llama/Meta-Llama-3-8B & Meta-Llama-3-8B
        try:
            model_config = Config.from_name(model_name)
        except ValueError:
            print(f"Model name {model_name} is not supported.\n")
            available_models = "\n".join(sorted(name_to_config))
            print(f"Available values:\n{available_models}")
            quit()

    hparams = capture_hparams()
    data = TinyLlama() if data is None else data

    config = Config.from_name(model_name) if model_config is None else model_config
    precision = precision or get_default_supported_precision(training=True)
    devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)
    # in case the dataset requires the Tokenizer
    tokenizer = Tokenizer(tokenizer_dir) if tokenizer_dir is not None else None

    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"pretrain-{config.name}",
        resume=bool(resume),
        log_interval=train.log_interval,
        log_args=asdict(log),
    )

    if devices * num_nodes > 1:
        strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy="HYBRID_SHARD")
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, num_nodes=num_nodes, strategy=strategy, precision=precision, loggers=[logger])

    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch()

    fabric.print(pprint.pformat(hparams))
    if logger_name in ("tensorboard", "wandb", "mlflow"):
        fabric.logger.log_hyperparams(hparams)

    main(
        fabric=fabric,
        devices=devices,
        num_nodes=num_nodes,
        seed=seed,
        initial_checkpoint_dir=initial_checkpoint_dir,
        resume=resume,
        config=config,
        data=data,
        out_dir=out_dir,
        tokenizer_dir=tokenizer_dir,
        tokenizer=tokenizer,
        train=train,
        eval=eval,
        kres=kres,
        optimizer=optimizer,
    )


def main(
    fabric: L.Fabric,
    devices: int,
    seed: int,
    initial_checkpoint_dir: Path | None,
    resume: bool | Literal["auto"] | Path,
    config: Config,
    data: DataModule,
    out_dir: Path,
    tokenizer_dir: Path | None,
    tokenizer: Tokenizer | None,
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: str | dict,
    num_nodes: int = 1,
    kres: KresArgs = KresArgs(),
) -> None:
    validate_args(train, eval, initial_checkpoint_dir, resume)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)

    initialize_weights(fabric, model, n_layer=config.n_layer, n_embd=config.n_embd)

    if train.tie_embeddings:
        model.transformer.wte.weight = model.lm_head.weight
    if train.max_seq_length:
        model.max_seq_length = train.max_seq_length

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(model):,}")

    model = torch.compile(model)
    model = fabric.setup(model)

    extra_kwargs = {"fused": fabric.device.type == "cuda"}
    optimizer = instantiate_torch_optimizer(optimizer, model.parameters(), **extra_kwargs)
    optimizer = fabric.setup_optimizers(optimizer)

    train_dataloader, eval_dataloaders, final_eval_dataloaders = get_dataloaders(
        fabric, data, tokenizer, train, model.max_seq_length
    )

    # 混合 dataloader 的交错比例是 replay/(new+replay)，只有总量恰好等于 plan 算的
    # train_max_tokens 时，跑完全程消耗的 replay 才恰好是一个 epoch。填小了 replay 段
    # 跑不满（尾巴那批文档只进了 C、没被重放），填大了 replay 中途耗尽。两种都不会自己
    # 暴露出来，所以在这里就硬失败。
    expected = getattr(data, "expected_max_tokens", None)
    if expected is not None and train.max_tokens != expected:
        raise SystemExit(
            f"--train.max_tokens={train.max_tokens} 与 replay_plan.json 里这条臂的 "
            f"train_max_tokens={expected} 不符。这个值必须照抄 plan：它等于「新域预算 + 该臂的 "
            f"replay 量」，直接填 8B 会让各臂的新域预算不相等。"
        )

    # setup_dataloaders 只接位置参数，所以先把名字抽出来、setup 完再按同样的顺序装回去。
    # 传单个参数时它返回的是 loader 本身而不是列表，这里始终至少有 train + val 两个
    _eval_names = list(eval_dataloaders) + list(final_eval_dataloaders)
    _n_periodic = len(eval_dataloaders)
    train_dataloader, *_eval_setup = fabric.setup_dataloaders(
        train_dataloader, *eval_dataloaders.values(), *final_eval_dataloaders.values()
    )
    eval_dataloaders = dict(zip(_eval_names[:_n_periodic], _eval_setup[:_n_periodic]))
    final_eval_dataloaders = dict(zip(_eval_names[_n_periodic:], _eval_setup[_n_periodic:]))

    if initial_checkpoint_dir:
        fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)

    state = {
        "model": model,
        "optimizer": optimizer,
        "train_dataloader": train_dataloader,
        "iter_num": 0,
        "step_count": 0,
    }

    resume = find_resume_path(resume, out_dir)
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    # W₀ 必须在权重就位之后才取，而"权重就位"有两条路：首段是上面的 load_raw，
    # 续跑是 fabric.load(resume, state)。放在这里两条路都已经走完。
    regularizer = build_regularizer(fabric, model, config, kres, initial_checkpoint_dir, bool(resume))

    train_time = time.perf_counter()

    # work around PyTorch issue https://github.com/pytorch/pytorch/issues/152162
    # which does not like the lazy initialization to be called in dynamo.
    # TODO: Happens with PyTorch 2.7+
    if (
        (_TORCH_EQUAL_2_7 or _TORCH_EQUAL_2_8)
        and (model._forward_module.__class__.__name__ == "OptimizedModule")
        and (model._forward_module._orig_mod.__class__.__name__ == "FullyShardedDataParallel")
    ):
        from torch.distributed.fsdp._runtime_utils import _root_pre_forward

        _root_pre_forward(model._forward_module._orig_mod, model._forward_module._orig_mod, [], {})

    fit(
        fabric=fabric,
        devices=devices,
        num_nodes=num_nodes,
        state=state,
        train_dataloader=train_dataloader,
        eval_dataloaders=eval_dataloaders,
        final_eval_dataloaders=final_eval_dataloaders,
        out_dir=out_dir,
        tokenizer_dir=tokenizer_dir,
        train=train,
        eval=eval,
        kres=kres,
        regularizer=regularizer,
    )

    # Save final checkpoint
    save_checkpoint(fabric, state, tokenizer_dir, out_dir / "final" / "lit_model.pth")

    total_tokens = state["iter_num"] * train.micro_batch_size * model.max_seq_length * fabric.world_size

    # Print formatted output
    separator = "-" * 40
    fabric.print(separator)
    fabric.print("| Performance")
    fabric.print(f"| - Total tokens  : {total_tokens:,}")
    fabric.print(f"| - Training Time : {(time.perf_counter() - train_time):.2f} s")
    fabric.print(f"| - Tok/sec       : {total_tokens / train_time:.2f} tok/s")
    fabric.print("| " + "-" * 40)

    if fabric.device.type == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        fabric.print("| Memory Usage")
        fabric.print(f"| - Memory Used   : {memory_used:.2f} GB")
    fabric.print(separator)


def build_regularizer(
    fabric: L.Fabric,
    model: nn.Module,
    config: Config,
    kres: KresArgs,
    initial_checkpoint_dir: Path | None,
    resuming: bool,
):
    """构造 OneReplay 协方差正则器，或在未启用时返回 None。

    这里是三个静默失效风险中两个的落点，所以每一步都硬失败、不给"降级继续"的路：

    1. **调用时机。** 必须在权重就位之后。首段靠 `load_raw`（上面 L223），续跑靠
       `fabric.load(resume, state)`，两者互斥，所以本函数只能在两条路都走完之后调。
       首段会拿 `lit_model.pth` 逐比特验证快照，这同时证明了模型确实收到了基座权重。

    2. **目标层清单不从 C 文件推。** 从模型的模块树上独立点出来，再用 `assert_covers`
       和 C 的键集双向比对。如果反过来拿 C 的键当清单，`assert_covers` 就变成自证，
       C 少了几层这件事永远查不出来——而那正是"惩罚静默变小、λ 悄悄换了含义"的入口。

    3. **续跑的 W₀ 来源。** 见 `KresArgs.base_checkpoint`：续跑段的模型权重已经漂移，
       W₀ 只能从基座文件读。
    """
    if not kres.enabled:
        return None

    from kres.covariance import module_in_features, target_names_from_linears
    from kres.regularizer import ReplayRegularizer, canonical_name

    if kres.base_checkpoint:
        base_checkpoint = Path(kres.base_checkpoint)
    elif initial_checkpoint_dir is not None:
        base_checkpoint = initial_checkpoint_dir / "lit_model.pth"
    else:
        raise ValueError(
            "启用了 `--kres.cov_path` 但没有 W₀ 的来源。续跑时 `--initial_checkpoint_dir` "
            "不可用（与 `--resume` 互斥），必须显式给 `--kres.base_checkpoint`，"
            "否则 W₀ 会变成当前漂移过的权重、惩罚方向随之改变"
        )

    linear_names = [canonical_name(n) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    target_names = target_names_from_linears(linear_names, config.n_layer, kres.include_lm_head)

    regularizer = ReplayRegularizer.from_path(
        kres.cov_path,
        model=model,
        target_names=target_names,
        checkpoint_path=base_checkpoint,
        device=fabric.device,
        identity=kres.identity,
        dtype=getattr(torch, kres.cov_dtype),
        resuming=resuming,
        module_shapes=module_in_features(model, target_names),
        reg_impl=kres.reg_impl,
        log_grad_norms=kres.log_grad_norms,
    )

    fabric.print(
        f"kres: λ={kres.replay_lambda} 目标层={len(target_names)} "
        f"identity={int(kres.identity)} lm_head={int(kres.include_lm_head)} "
        f"注入点={'clip 之前' if kres.inject_before_clip else 'clip 之后'} "
        f"W₀来源={'基座文件（续跑）' if resuming else '模型快照（已逐比特验证）'}"
    )
    fabric.print(
        f"kres: C {regularizer.memory_bytes() / 1e9:.2f} GB + "
        f"W₀ {regularizer.reference_memory_bytes() / 1e9:.2f} GB 常驻（成本表用这两个数）"
    )
    return regularizer


def apply_penalty(fabric: L.Fabric, regularizer, model: nn.Module, kres: KresArgs) -> tuple[float, dict]:
    """把 λ·dR/dW 累加进 .grad，返回 (归一化后的 R, 统计量)。

    两条实现算的是同一个惩罚，都**每个 optimizer step 只调一次**：

    - analytic 直接写 `.grad`，是正式路径。
    - autograd 对 λ·R 调 backward，梯度同样累加进 `.grad`。它只为等价性校验存在，
      所以这里不去拆出惩罚自己的范数（要拆得先把 `.grad` 存一份，那是 1.41 GB 的
      临时开销，不值得为一条校验路径付）。`reg_grad_norm` 因此报 nan。
    """
    if regularizer.injects_grad:
        return regularizer.accumulate_grad(model, kres.replay_lambda)

    from kres.regularizer import global_grad_norm

    lm_norm = global_grad_norm(model) if kres.log_grad_norms else float("nan")
    penalty = regularizer.penalty(model)
    fabric.backward(penalty * kres.replay_lambda)
    stats = {
        "used_layers": float(len(regularizer.layers(model))),
        "lm_grad_norm": lm_norm,
        "reg_grad_norm": float("nan"),
        "total_grad_norm": global_grad_norm(model) if kres.log_grad_norms else float("nan"),
    }
    return float(penalty), stats


def fit(
    fabric: L.Fabric,
    devices: int,
    state: dict,
    train_dataloader: DataLoader,
    eval_dataloaders: dict[str, DataLoader],
    final_eval_dataloaders: dict[str, DataLoader],
    out_dir: Path,
    tokenizer_dir: Path | None,
    train: TrainArgs,
    eval: EvalArgs,
    num_nodes: int = 1,
    kres: KresArgs = KresArgs(),
    regularizer=None,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]

    # 训练日志那行 "val: x.xxx" 报的是哪个集合。第一个就是主验证集（新域 held-out）
    primary_eval = next(iter(eval_dataloaders), None)

    # 曲线的横轴。**跨臂比较不能用 step**：各臂 step 里掺的 replay 比例不同，同样是第
    # 200 步，vanilla 已经吃了 419M 新域 token，replay 8B 只吃了约 210M，另一半是重放。
    # 按 step 对齐等于拿「学了两倍新域」的点去比「忘得更少」。
    # 取 _dataloader：_FabricDataLoader 只 update 了实例字典，属性是类上的，不转发
    _train_inner = getattr(train_dataloader, "_dataloader", train_dataloader)

    def x_axis() -> dict[str, float]:
        consumed = getattr(_train_inner, "new_tokens_consumed", None)
        return {} if consumed is None else {"new_tokens": float(consumed)}

    if eval.initial_validation:
        # step 0 这次评的就是**基座本身**（权重刚从 initial_checkpoint_dir 载入、还没
        # 更新过）。probe 上的这个数是遗忘量的基准，没有它，后面所有 probe loss 都只是
        # 一个孤立的绝对值，说不出"掉了多少"。零成本，正式训练一定要开
        losses = evaluate_all(fabric, model, eval_dataloaders, eval.max_iters)
        fabric.print(f"Initial evaluation | {format_losses(losses)}")
        fabric.log_dict({**loss_metrics(losses), **x_axis()}, step=state["iter_num"])
        val_loss = f"{losses[primary_eval]:.3f}" if primary_eval else "n/a"
    else:
        fabric.print("Verifying settings ...")
        for dl in eval_dataloaders.values():
            validate(fabric, model, dl, max_iters=2, verbose=False)  # sanity check
        val_loss = "n/a"

    throughput = ThroughputMonitor(fabric, window_size=5)

    with torch.device("meta"):
        meta_model = GPT(model.config)
        x = torch.randint(0, 1, (train.micro_batch_size, meta_model.max_seq_length))
        model_fwd = lambda: meta_model(x)  # noqa: F821
        model_loss = lambda y: chunked_cross_entropy(y, x, chunk_size=0)  # noqa: F821
        measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    max_tokens_per_device = train.max_tokens // fabric.world_size
    tokens_per_iter = train.micro_batch_size * model.max_seq_length
    max_iters = max_tokens_per_device // tokens_per_iter
    log_iter_interval = train.log_interval * train.gradient_accumulation_iters(devices, num_nodes)
    initial_iter = state["iter_num"]
    train_iterator = CycleIterator(train_dataloader)

    running_loss = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=False).to(
        fabric.device
    )
    # 最近一个 optimizer step 的惩罚量 + 裁剪触发计数。裁剪触发率是判断
    # `inject_before_clip` 到底有没有实际影响的唯一依据：它一直是 0 的话，
    # 注入点选哪边完全等价；接近 1 的话，λ 的隐式缩放就是逐步在发生的。
    # 基线臂（没有正则器）也要记这两个数，否则正则臂那个触发率没有分母：loss spike
    # 本来就会把范数顶过 max_norm，不减掉基线就没法把触发归因到惩罚头上。
    # window_* 是上一个 log 区间内的计数，累计率会把 warmup 的暂态一直摊进后面，
    # 分不清裁剪是开头几步的事还是全程都在发生。
    reg_running = {"R": 0.0, "clip_hits": 0, "clip_steps": 0, "window_hits": 0, "window_steps": 0}

    # 计时器与 kres 正则无关地建起来：成本表要对比三条臂，baseline 臂也得有同一套口径的
    # 数，否则"贵多少"就没有分母。只受 --kres.profile 控制
    from kres.profiling import PhaseTimer, cost_record, format_cost_summary, reset_peak_memory
    from kres.regularizer import global_grad_norm

    timer = PhaseTimer(fabric.device, enabled=kres.profile, warmup_steps=kres.profile_warmup_steps)
    peaks_reset = False
    # steady 段的起点：时间与 step 数必须成对推进，否则 sec_per_step 的分子分母口径不一致
    steady_step0 = 0

    fabric.barrier()
    total_t0 = time.perf_counter()
    steady_t0 = total_t0

    warmup_iters = train.warmup_iters(devices, num_nodes, max_iters, train_dataloader)
    # cooldown 锚在本臂自己的 max_iters 上，而 max_iters 来自本臂的 --train.max_tokens。
    # 各臂总步数不同（新域 8B 之外还要过各自的 replay），这样每条臂都完整走一遍
    # warmup→stable→cooldown，终点都是退火完成的状态
    cooldown_iters = int(max_iters * train.lr_cooldown_fraction) if train.lr_schedule == "wsd" else 0
    ga = train.gradient_accumulation_iters(devices, num_nodes)
    # 这个检查不能只在 rank 0 上做：只有一个 rank 退出会把其余 rank 挂在下一次 barrier 上
    if warmup_iters + cooldown_iters > max_iters:
        raise SystemExit(
            f"warmup({warmup_iters // ga}) + cooldown({cooldown_iters // ga}) 超过总步数 "
            f"{max_iters // ga}，没有 stable 段。调小 --train.lr_warmup_steps 或 "
            f"--train.lr_cooldown_fraction。"
        )
    fabric.print(
        f"LR：{train.lr_schedule} peak={optimizer.defaults['lr']:.2e} min={train.min_lr:.2e}，"
        f"共 {max_iters // ga} 个 optimizer step（"
        + (
            f"warmup {warmup_iters // ga} + stable {(max_iters - warmup_iters - cooldown_iters) // ga}"
            f" + cooldown {cooldown_iters // ga}"
            if train.lr_schedule == "wsd"
            else f"warmup {warmup_iters // ga} + cosine {(max_iters - warmup_iters) // ga}"
        )
        + "）"
    )

    # 原来是 `for train_data in train_iterator:` 再在循环体开头 `if iter_num >= max_iters: break`，
    # 也就是**先取一个 batch 再判断要不要用它**。跑满 max_iters 之后还会多取一次，只为了
    # 把它扔掉。上游那样写没问题：LitData 配 CycleIterator 是无限流，多取一个无所谓。
    # 但本项目的 replay 混合 dataloader 名额是按「每段恰好一个 epoch」精确算出来的，
    # 多要的那一个直接触发它的耗尽护栏——60 步全部跑完、checkpoint 也存了，然后在收尾
    # 处抛 RuntimeError，成本汇总和消耗对账都打不出来。改成先判断后取。
    # CycleIterator.__next__ 会惰性初始化内部迭代器，可以直接 next()。
    while state["iter_num"] < max_iters:
        train_data = next(train_iterator)

        # determine and set the learning rate for this iteration
        lr = get_lr(
            optimizer.defaults["lr"],
            state["iter_num"],
            warmup_iters,
            max_iters,
            train.min_lr,
            schedule=train.lr_schedule,
            cooldown_iters=cooldown_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        state["iter_num"] += 1
        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.max_seq_length].contiguous().long()
        targets = train_data[:, 1 : (model.max_seq_length + 1)].contiguous().long()

        is_accumulating = state["iter_num"] % train.gradient_accumulation_iters(devices, num_nodes) != 0
        with timer.track("fwd_bwd"), fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets)
            fabric.backward(loss / train.gradient_accumulation_iters(devices, num_nodes))

        running_loss.update(loss.detach())

        if not is_accumulating:
            # 惩罚每个 optimizer step 只注入一次。R 只依赖 W 和 C，而 W 在一个梯度
            # 累积窗口内不变（optimizer.step 只在窗口末尾执行），所以窗口内它是常量。
            # 放进上面的 micro-batch 循环就是 gradient_accumulation_iters 倍的 λ
            # （本项目约 125，取决于 global_batch_size），而这个错误不报任何异常。
            if regularizer is not None and kres.inject_before_clip:
                with timer.track("penalty"):
                    reg_value, reg_stats = apply_penalty(fabric, regularizer, model, kres)

            # 测点必须在 clip 之前，而且要用 clip_gradients 实际看到的那个范数：
            # inject_before_clip 时那是注入之后的总范数，apply_penalty 刚算过，别再算一遍；
            # 其余三种情形（基线臂、或先裁后注入）clip 看到的都是纯 LM 梯度的范数
            if kres.log_grad_norms and train.max_norm:
                injected = regularizer is not None and kres.inject_before_clip
                pre_clip_norm = reg_stats["total_grad_norm"] if injected else global_grad_norm(model)
                if regularizer is None:
                    # 有正则器时这个字段归 apply_penalty 的 reg_stats 管，下面的 update 会覆盖，
                    # 别在这里跟它抢；基线臂没人写，只能在这里填
                    reg_running["lm_grad_norm"] = pre_clip_norm
                if pre_clip_norm > train.max_norm:
                    reg_running["clip_hits"] += 1
                    reg_running["window_hits"] += 1
                reg_running["clip_steps"] += 1
                reg_running["window_steps"] += 1

            with timer.track("clip"):
                fabric.clip_gradients(model, optimizer, max_norm=train.max_norm)

            if regularizer is not None and not kres.inject_before_clip:
                with timer.track("penalty"):
                    reg_value, reg_stats = apply_penalty(fabric, regularizer, model, kres)

            if regularizer is not None:
                reg_running["R"] = reg_value
                reg_running.update(reg_stats)

            with timer.track("optimizer"):
                optimizer.step()
                optimizer.zero_grad()
            state["step_count"] += 1
            timer.mark_step()

            # warmup 一过就重开显存窗口。编译期的临时峰值可能比 steady state 还高，
            # 不重置的话成本表报的是编译峰值，而那个数与训练本身无关
            if not peaks_reset and not timer.in_warmup:
                reset_peak_memory([fabric.device])
                steady_t0 = time.perf_counter()
                steady_step0 = state["step_count"]
                peaks_reset = True

        if state["iter_num"] % log_iter_interval == 0:
            loss = running_loss.compute().item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=(t1 - total_t0),
                flops=(measured_flops * log_iter_interval),
                batches=state["iter_num"],
                samples=(state["iter_num"] * train.micro_batch_size),
                lengths=(state["iter_num"] * train.micro_batch_size * model.max_seq_length),
            )
            metrics = {
                "loss": loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "remaining_time": (
                    (t1 - total_t0) / (state["iter_num"] - initial_iter) * (max_iters - state["iter_num"])
                ),
                "tokens": state["iter_num"] * train.micro_batch_size * model.max_seq_length,
                "total_tokens": (state["iter_num"] * train.micro_batch_size * model.max_seq_length * fabric.world_size),
                "learning_rate": lr,
                # total_tokens 把 replay 也算进去了，跨臂对齐要用新域那份，见 x_axis
                **x_axis(),
            }
            if kres.log_grad_norms and train.max_norm:
                metrics.update(
                    {
                        "lm_grad_norm": reg_running.get("lm_grad_norm", float("nan")),
                        "clip_rate": reg_running["clip_hits"] / max(reg_running["clip_steps"], 1),
                        "clip_rate_window": reg_running["window_hits"] / max(reg_running["window_steps"], 1),
                    }
                )
                reg_running["window_hits"] = reg_running["window_steps"] = 0
            if regularizer is not None:
                metrics.update(
                    {
                        "reg_R": reg_running["R"],
                        "reg_lambda": kres.replay_lambda,
                        "reg_used_layers": reg_running.get("used_layers", 0.0),
                        "reg_grad_norm": reg_running.get("reg_grad_norm", float("nan")),
                        "total_grad_norm": reg_running.get("total_grad_norm", float("nan")),
                    }
                )
            if isinstance(val_loss, float):
                val_loss = f"{val_loss:.3f}"
            # 字段顺序别动：02_smoke_pretrain.pbs 的汇总按这个顺序做正则。基线臂没有 R
            # 和 |g_reg|，但 |g_lm| 与 clip 要用同样的格式打出来——正则臂那个触发率得减掉
            # 基线才能归因到惩罚上。clip 打两个数：累计率 / 上一个 log 区间内的率。
            if regularizer is not None:
                extra = (
                    f" | R: {metrics['reg_R']:.3e}"
                    f" |g_lm|: {metrics['lm_grad_norm']:.3f}"
                    f" |g_reg|: {metrics['reg_grad_norm']:.3e}"
                    f" clip: {metrics['clip_rate']:.0%}/{metrics['clip_rate_window']:.0%}"
                )
            elif "clip_rate" in metrics:
                extra = (
                    f" | |g_lm|: {metrics['lm_grad_norm']:.3f}"
                    f" clip: {metrics['clip_rate']:.0%}/{metrics['clip_rate_window']:.0%}"
                )
            else:
                extra = ""
            fabric.print(
                f"Epoch {metrics['epoch'] + 1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" val: {val_loss} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
                f" remaining time: {timedelta(seconds=int(metrics['remaining_time']))!s}" + extra
            )

            throughput_metrics = throughput.compute()
            metrics.update(throughput_metrics)
            fabric.log_dict(metrics, step=state["iter_num"] - 1)

        if eval_dataloaders and not is_accumulating and state["step_count"] % eval.interval == 0:
            t0 = time.perf_counter()
            losses = evaluate_all(fabric, model, eval_dataloaders, eval.max_iters)
            td = time.perf_counter() - t0
            val_loss = losses[primary_eval]

            fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: "
                f"{format_losses(losses)} | eval time: {td * 1000:.2f} ms"
            )
            fabric.log_dict({**loss_metrics(losses), **x_axis()}, step=state["iter_num"] - 1)
            fabric.barrier()

        if train.save_interval is not None and not is_accumulating and state["step_count"] % train.save_interval == 0:
            save_checkpoint(fabric, state, tokenizer_dir, out_dir / f"step-{state['step_count']:08d}" / "lit_model.pth")

    # 成本汇总。分母用 steady_t0 而不是 total_t0，也就是 warmup 之后那一段——per-step
    # 时间要跨臂比较，掺进编译时间就不可比了。只在 rank 0 打印，否则多卡会刷 world_size 份
    # step 数从 steady 段起点算起，和 steady_t0 配对。跑得比 warmup 还短时 steady_step0
    # 仍是 0，此时报的就是含 warmup 的全程——短 run 本来就没有 steady state 可言
    steady_steps = max(state["step_count"] - steady_step0, 0)
    record = cost_record(
        [fabric.device],
        grad_accum=train.gradient_accumulation_iters(devices, num_nodes),
        train_sec=time.perf_counter() - steady_t0,
        steps=steady_steps,
        tokens=steady_steps
        * train.gradient_accumulation_iters(devices, num_nodes)
        * train.micro_batch_size
        * model.max_seq_length
        * fabric.world_size,
        timer=timer,
        covariance_bytes=regularizer.memory_bytes() if regularizer is not None else 0,
        reference_bytes=regularizer.reference_memory_bytes() if regularizer is not None else 0,
        extra={"reg_lambda": kres.replay_lambda, "reg_enabled": float(regularizer is not None)},
    )
    if fabric.global_rank == 0:
        fabric.print(format_cost_summary(record))
    fabric.log_dict({k: v for k, v in record.items() if isinstance(v, (int, float))}, step=state["iter_num"])

    # 混合 dataloader 的收尾对账：新域和 replay 各真的喂了多少。replay 没跑满一个 epoch
    # 的话，没被重放的那批文档仍然进了 C，「replay 与 C 同源」对它们就不成立了——这件事
    # 除了在这里报，没有别的地方看得出来
    # 取 _dataloader：_FabricDataLoader 只 update 了实例字典，方法是类上的，不转发
    summary = getattr(getattr(train_dataloader, "_dataloader", train_dataloader), "consumption_summary", None)
    if summary is not None:
        fabric.print(summary())

    # Final validation。这里才把 test 加进来：它全程不评，就是为了不参与选 λ
    if eval.final_validation:
        losses = evaluate_all(fabric, model, {**eval_dataloaders, **final_eval_dataloaders}, eval.max_iters)
        fabric.log_dict({**loss_metrics(losses), **x_axis()}, step=state["iter_num"])
        fabric.print(f"Final evaluation | {format_losses(losses)}")


def evaluate_all(
    fabric: L.Fabric, model: nn.Module, loaders: dict[str, DataLoader], max_iters: int
) -> dict[str, float]:
    """在每个评估集上各跑一遍，返回 {名字: loss}。"""
    return {
        name: validate(fabric, model, dl, max_iters=max_iters, verbose=False).item()
        for name, dl in loaders.items()
    }


def format_losses(losses: dict[str, float]) -> str:
    """loss 与 ppl 并排打。

    ppl = exp(loss) 是同一个数的单调变换，没有额外信息量，但论文报的是 ppl，直接打出来
    省得事后换算、也省得换算时把底数记错。它等于**语料级**困惑度而不是「各 batch ppl 的
    平均」，因为 `validate` 平均的是各 batch 的 loss，而预训练数据是打包过的、每个 batch
    恰好 micro_batch × seq_len 个 token 且没有 ignore_index，无权平均正好等于 token 加权
    平均。哪天评估集改成带 padding 的，这个等式就不成立了，得改成按 token 数加权。
    """
    return " | ".join(f"{name} loss {v:.4f} ppl {math.exp(v):.3f}" for name, v in losses.items())


def loss_metrics(losses: dict[str, float]) -> dict[str, float]:
    """展平成 logger 的列名：val_loss / val_ppl / probe_loss / probe_ppl / ..."""
    metrics = {}
    for name, v in losses.items():
        metrics[f"{name}_loss"] = v
        metrics[f"{name}_ppl"] = math.exp(v)
    return metrics


@torch.no_grad()
def validate(
    fabric: L.Fabric, model: nn.Module, val_dataloader: DataLoader, max_iters: int, verbose: bool = True
) -> torch.Tensor:
    fabric.barrier()
    if verbose:
        fabric.print("Validating ...")
    model.eval()

    losses = []
    for k, batch in enumerate(val_dataloader):
        if k >= max_iters:
            break
        input_ids = batch[:, 0 : model.max_seq_length].contiguous().long()
        targets = batch[:, 1 : (model.max_seq_length + 1)].contiguous().long()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets)
        losses.append(loss)

    val_loss = torch.stack(losses).mean()
    model.train()
    fabric.barrier()
    return val_loss


def get_dataloaders(
    fabric: L.Fabric, data: DataModule, tokenizer: Tokenizer, train: TrainArgs, block_size: int
) -> tuple[DataLoader, dict[str, DataLoader], dict[str, DataLoader]]:
    """返回 (训练流, 周期性评估集, 只在收尾评一次的集合)。

    上游的 DataModule 只有 `val_dataloader()` 一个验证集，对它们两个 dict 退化成
    `{"val": ...}` 和 `{}`，行为与改动前完全一致。`ReplayMixedData` 额外给出 probe
    （旧域 held-out，遗忘量）和 test（新域 held-out，不参与调参）。
    """
    data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, max_seq_length=block_size)
    with fabric.rank_zero_first():
        data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()

    if hasattr(data, "eval_dataloaders"):
        eval_dataloaders = dict(data.eval_dataloaders())
    else:
        eval_dataloaders = {"val": data.val_dataloader()}
    final_eval_dataloaders = dict(data.final_eval_dataloaders()) if hasattr(data, "final_eval_dataloaders") else {}

    # 同名会让收尾那次的 log_dict 用同一个键写两遍，后写的静默覆盖先写的，
    # 而两者评的是不同数据
    overlap = sorted(set(eval_dataloaders) & set(final_eval_dataloaders))
    if overlap:
        raise SystemExit(f"评估集重名：{overlap} 同时出现在周期性评估和收尾评估里")
    return train_dataloader, eval_dataloaders, final_eval_dataloaders


def get_lr(
    learning_rate: float,
    it: int,
    warmup_iters: int,
    max_iters: int,
    min_lr: float,
    schedule: str = "cosine",
    cooldown_iters: int = 0,
) -> float:
    """两种 schedule：cosine（litgpt 原逻辑）与 wsd（warmup-stable-decay）。

    `it` 与 `max_iters` 都以 micro-batch 计（调用点传的是 `state["iter_num"]`），不是
    optimizer step。两者同口径，所以形状是对的，但读日志时别把 warmup_iters 当成步数。

    **wsd 的 cooldown 锚在本臂自己的 max_iters 上**，而 max_iters 由本臂的
    `--train.max_tokens` 推出。三条 replay 臂总步数不同（新域 8B 之外还要过 1/4/8B 的
    replay），这样每条臂都完整走一遍 warmup→stable→cooldown，终点都是退火完成的状态，
    臂间才可比。若改成共用一条绝对曲线，短的那条臂会在 stable 段中途被切断，它的终点
    模型是「还热着」的，与退火完的臂比 loss 等于在比两种不同的东西。
    """
    # 1) 线性 warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr

    if schedule == "wsd":
        if cooldown_iters <= 0:
            return learning_rate
        # 2) stable：cooldown 开始之前一直保持 peak
        decay_start = max_iters - cooldown_iters
        if it < decay_start:
            return learning_rate
        # 3) 线性 cooldown。+1 是为了让最后一个 iter 正好落在 min_lr 上——退火的终点值
        #    直接决定终态模型，差一格就不是「退火完成」了
        ratio = min((it - decay_start + 1) / cooldown_iters, 1.0)
        return learning_rate - ratio * (learning_rate - min_lr)

    # cosine（litgpt 原逻辑）
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def initialize_weights(fabric: L.Fabric, model: GPT, n_layer: int, n_embd: int) -> None:
    """GPT-NeoX weight initialization (https://arxiv.org/abs/2204.06745)."""
    # Adapted from https://github.com/jzhang38/TinyLlama

    def init_weights(module, std):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    for mod in model.modules():
        if isinstance(mod, (nn.Embedding, nn.Linear)):
            mod.reset_parameters = partial(init_weights, mod, std=math.sqrt(2.0 / 5 / n_embd))

    # need a separate loop because `mod.proj` below is a `nn.Linear` too
    for mod in model.modules():
        if isinstance(mod, (LLaMAMLP, CausalSelfAttention)):
            mod.proj.reset_parameters = partial(init_weights, mod.proj, std=(1 / math.sqrt(n_embd) / n_layer))

    if not isinstance(fabric.strategy, FSDPStrategy):
        reset_parameters(model)


def save_checkpoint(fabric, state, tokenizer_dir, checkpoint_file):
    model = state["model"]
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Saving checkpoint to {str(checkpoint_file)!r}")
    fabric.save(checkpoint_file, state)
    if fabric.global_rank == 0:
        save_hyperparameters(setup, checkpoint_file.parent)
        if tokenizer_dir is not None:
            copy_config_files(tokenizer_dir, checkpoint_file.parent)
        save_config(model.config, checkpoint_file.parent)


def validate_args(train: TrainArgs, eval: EvalArgs, initial_checkpoint_dir, resume) -> None:
    issues = []
    unsupported = [(train, ["epochs"]), (eval, ["max_new_tokens"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    # 这份 fit() 的循环只看 `iter_num >= max_iters`（由 max_tokens 推出），max_steps
    # 一次都没被读过。原来只发一句「建议用于 profiling/debug」的警告，等于承诺它有效——
    # 拿它跑 50 步冒烟的人会得到一个跑满 3814 步的作业，而警告已经在几千行日志之前滚走了。
    # 要限制步数就把 max_tokens 设成 步数 × global_batch_size × seq_len。
    if train.max_steps is not None:
        issues.append(
            "`--train.max_steps` 在这份 pretrain.py 里不生效（循环只看由 max_tokens 推出的 "
            "max_iters）。要跑短 run，请设 --train.max_tokens = 步数 × global_batch_size × seq_len。"
        )
    required = [(train, ["max_tokens", "max_norm"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if initial_checkpoint_dir and resume:
        issues.append("Can't provide both `--resume` and `--initial_checkpoint_dir`. Choose one.")
    if issues:
        raise ValueError("\n".join(issues))
