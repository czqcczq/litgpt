# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import math
import warnings
from dataclasses import dataclass


@dataclass
class TrainArgs:
    """Training-related arguments"""

    save_interval: int | None = 1000
    """Number of optimizer steps between saving checkpoints"""
    log_interval: int = 1
    """Number of iterations between logging calls"""
    global_batch_size: int = 64
    """Number of samples between optimizer steps across data-parallel ranks"""
    micro_batch_size: int = 4
    """Number of samples per data-parallel rank"""
    lr_warmup_steps: int | None = 100
    """Number of iterations with learning rate warmup active"""
    lr_warmup_fraction: float | None = None
    """The fraction of an epoch to use for learning rate warmup"""
    epochs: int | None = None
    """Number of epochs to train on"""
    # TODO: `pretrain` is the only script using `max_tokens` explicitly. replace it with epoch_size*epochs?
    max_tokens: int | None = None
    """Total number of tokens to train on"""
    max_steps: int | None = None
    """Limits the number of optimizer steps to run"""
    max_time: float | None = None
    """Limits the number of seconds to train for"""
    max_seq_length: int | None = None
    """Limits the length of samples"""
    tie_embeddings: bool | None = None
    """Whether to tie the embedding weights with the language modeling head weights"""

    # Optimization args
    max_norm: float | None = None
    min_lr: float = 6e-5

    def __post_init__(self) -> None:
        if self.lr_warmup_fraction and self.lr_warmup_steps:
            raise ValueError(
                "Can't provide both `--train.lr_warmup_fraction` and `--train.lr_warmup_steps`. Choose one."
            )
        if self.lr_warmup_fraction and not (0 <= self.lr_warmup_fraction <= 1):
            raise ValueError("`--train.lr_warmup_fraction` must be between 0 and 1.")

        if self.lr_warmup_steps and self.max_steps and (self.lr_warmup_steps >= self.max_steps):
            warnings.warn(
                "`--train.lr_warmup_steps` should be less than `--train.max_steps`."
                f" Got {self.lr_warmup_steps} lr_warmup_steps and {self.max_steps} max_steps.",
                UserWarning,
            )

    def gradient_accumulation_iters(self, devices: int, num_nodes: int = 1) -> int:
        """Number of iterations between gradient synchronizations"""
        gradient_accumulation_iters = self.batch_size(devices, num_nodes) // self.micro_batch_size
        assert gradient_accumulation_iters > 0
        return gradient_accumulation_iters

    def batch_size(self, devices: int, num_nodes: int = 1) -> int:
        """Number of samples between optimizer steps per data-parallel rank"""
        batch_size = self.global_batch_size // (devices * num_nodes)
        assert batch_size > 0
        return batch_size

    def warmup_iters(self, devices: int, num_nodes: int, max_iters: int, train_dataloader) -> int:
        """Number of iterations to warm up the learning rate."""
        if self.lr_warmup_fraction:
            return min(max_iters, math.ceil(self.lr_warmup_fraction * len(train_dataloader)))
        if self.lr_warmup_steps:
            return min(max_iters, self.lr_warmup_steps * self.gradient_accumulation_iters(devices, num_nodes))
        return 0


@dataclass
class EvalArgs:
    """Evaluation-related arguments"""

    interval: int = 600
    """Number of optimizer steps between evaluation calls"""
    max_new_tokens: int | None = None
    """Number of tokens to generate"""
    max_iters: int = 100
    """Number of iterations"""
    initial_validation: bool = False
    """Whether to evaluate on the validation set at the beginning of the training"""
    final_validation: bool = True
    """Whether to evaluate on the validation set at the end of the training"""
    evaluate_example: str | int = "first"
    """How to pick an example instruction to evaluate periodically during training.
       Can be "first", "random", or an integer index to pick a specific example."""


@dataclass
class KresArgs:
    """OneReplay 协方差正则的参数（k-res 新增，不属于上游 LitGPT）。

    `cov_path` 为 None 时整个特性关闭，训练路径与上游完全一致——No Replay 和
    Vanilla Replay 两条臂就是这么跑的，不需要另一份 pretrain.py。
    """

    cov_path: str | None = None
    """采好的 C 文件（kres.collect_cov 的产物）。None 表示不加正则"""
    replay_lambda: float = 0.0
    """惩罚权重。R = (λ/N) Σ_l tr(ΔW_l C_l ΔW_lᵀ)，N 是实际解析到的层数"""
    base_checkpoint: str | None = None
    """W₀ 的来源 lit_model.pth。

    不能用 `initial_checkpoint_dir` 代替：那个参数与 `--resume` 互斥
    （pretrain.py 的 validate_args），而 6 小时 walltime 下单臂要续跑 5 次。
    续跑时模型里的权重已经漂移了，W₀ 只能从这个文件读。留空则回退到
    `initial_checkpoint_dir / lit_model.pth`（仅首段可用）。
    """
    include_lm_head: bool = False
    """是否把 lm_head 纳入惩罚。必须与采 C 时的 --include-lm-head 一致"""
    identity: bool = False
    """把每个 C 换成同尺寸单位矩阵。惩罚退化成 ‖ΔW‖_F²，这是主消融臂"""
    inject_before_clip: bool = True
    """惩罚梯度注入在 clip_gradients 之前（True）还是之后（False）。

    两个选项保住的是不同的不变量，都不是"更对"：
    - True：裁剪对整个梯度向量做同一个标量缩放，所以 LM 与惩罚的**比例**被精确保住，
      λ 的语义干净。代价是 onereplay 臂因为总范数更大而被裁得更狠，等效学习率比
      baseline 臂小——而"学习率小则遗忘少"本身就是已知效应，会混进结论里。
    - False：LM 侧的等效学习率与 baseline 臂完全一致，但惩罚项被 1/c 放大，
      其中 c 是当步的裁剪系数，于是 λ 变成逐步波动的量。
    真正决定这件事有没有影响的是裁剪的触发率，所以两种模式下都记 clip_rate。
    """
    reg_impl: str = "analytic"
    """analytic 直接写 .grad；autograd 把 R 加到 loss 上。两条路算同一个量，
    后者只为等价性校验保留"""
    cov_dtype: str = "float32"
    """C 常驻的 dtype。float32 在 0.4b 基座上是 1.67 GB，bfloat16 减半"""
    log_grad_norms: bool = True
    """记 lm/reg/total 三个梯度范数。这是判断裁剪有没有实际影响的唯一依据"""
    profile: bool = False
    """开分阶段计时。要 torch.cuda.synchronize，会拖慢训练，所以只在专门测成本的 run 里开。

    显存峰值和成本汇总不受这个开关影响，一直都记——读 allocator 计数器是免费的。
    """
    profile_warmup_steps: int = 20
    """前多少个 optimizer step 只跑不记。

    LitGPT 默认 torch.compile，前几步是编译时间，混进均值会让 per-step 时间虚高数倍。
    成本对比只用 steady state 的数，否则"加了正则慢 30%"可能纯粹是两个 run 的编译时间差。
    """

    def __post_init__(self) -> None:
        if self.reg_impl not in ("analytic", "autograd"):
            raise ValueError(f"`--kres.reg_impl` 只能是 analytic 或 autograd，收到 {self.reg_impl!r}")
        if self.cov_path is None and self.replay_lambda:
            raise ValueError(
                "给了 `--kres.replay_lambda` 却没给 `--kres.cov_path`，惩罚不会生效。"
                "这种组合唯一的结果是跑出一条以为加了正则、实际是 baseline 的臂"
            )

    @property
    def enabled(self) -> bool:
        return self.cov_path is not None


@dataclass
class LogArgs:
    """Logging-related arguments. Different loggers use different fields."""

    # === WandB Fields ===
    project: str | None = None
    """WandB project name"""
    run: str | None = None
    """WandB run name (defaults to generated name)"""
    group: str | None = None
    """WandB group name"""

    # === LitLogger Fields (Lightning.ai) ===
    teamspace: str | None = None
    """Teamspace name where charts and artifacts will appear"""
    metadata: dict | None = None
    """Extra metadata to associate with the experiment as tags"""
    log_model: bool = False
    """If True, automatically log model checkpoints as artifacts"""
    save_logs: bool = True
    """If True, capture and upload terminal logs"""
    checkpoint_name: str | None = None
    """Override the base name for logged checkpoints"""
