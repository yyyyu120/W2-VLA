"""
Trainer for QwenSubtaskM2W.

Default:
  - Qwen-VL trainability is controlled by trainer.train_qwen_vl.
  - JEPA predictor, wrist context adapter, and action head are trainable.
  - Frozen wrist V-JEPA2.1 is used only as a latent teacher.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    normalize_dotlist_args,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = initialize_overwatch(__name__)


def bind_local_cuda_device() -> None:
    """Bind each distributed process before any early NCCL barrier."""
    if not torch.cuda.is_available():
        return
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    torch.cuda.set_device(int(local_rank))


def setup_directories(cfg) -> Path:
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
    return output_dir


def save_run_config(cfg, output_dir: Path) -> None:
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    config_save_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_save_path)
    logger.info(f"Saved run config: {config_save_path}")


def build_model(cfg):
    logger.info(f"Building QwenSubtaskM2W from `{cfg.framework.qwenvl.base_vlm}`")
    return build_framework(cfg)


def _is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("RANK", 0)) == 0


def _data_cfg_int(data_cfg, key: str, default: int) -> int:
    try:
        return int(data_cfg.get(key, default))
    except (AttributeError, TypeError, ValueError):
        return default


def prepare_data(cfg, accelerator):
    logger.info(f"Creating dataset with mix `{cfg.datasets.vla_data.data_mix}`")
    dataset_py = cfg.datasets.vla_data.dataset_py
    if dataset_py in {"subtask_m2w_datasets_w2", "subtask_m2w_robotwin_w2_datasets"}:
        if dataset_py == "subtask_m2w_robotwin_w2_datasets":
            from starVLA.dataloader.subtask_m2w_robotwin_w2_datasets import (
                collate_fn,
                get_vla_dataset,
            )
        else:
            from starVLA.dataloader.subtask_m2w_datasets_w2 import collate_fn, get_vla_dataset

        vla_dataset_cfg = cfg.datasets.vla_data
        num_workers = _data_cfg_int(vla_dataset_cfg, "num_workers", 4)
        prefetch_factor = _data_cfg_int(vla_dataset_cfg, "prefetch_factor", 2)
        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        dataloader = DataLoader(
            vla_dataset,
            batch_size=vla_dataset_cfg.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )
        if _is_main_process():
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
    else:
        dataloader = build_dataloader(cfg=cfg, dataset_py=dataset_py)
    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return dataloader


def build_optimizer(model, cfg):
    def params(module):
        return [parameter for parameter in module.parameters() if parameter.requires_grad]

    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 2.5e-5)
    jepa_lr = lr_cfg.get("jepa_predictor", base_lr * 2)
    wrist_context_lr = lr_cfg.get("wrist_context_adapter", jepa_lr)
    action_lr = lr_cfg.get("action_model", base_lr)
    vlm_lr = lr_cfg.get("vlm", base_lr * 0.2)
    train_qwen_vl = bool(cfg.trainer.get("train_qwen_vl", False))
    vlm_initial_lr = vlm_lr if train_qwen_vl else 0.0

    param_groups = [
        {"name": "jepa_predictor", "params": params(model.jepa_predictor), "lr": jepa_lr},
        {"name": "action_heads", "params": params(model.action_model), "lr": action_lr},
        {"name": "vlm_backbone", "params": params(model.qwen_vl_interface), "lr": vlm_initial_lr},
    ]
    wrist_context_adapter = getattr(model, "wrist_context_adapter", None)
    if wrist_context_adapter is not None:
        param_groups.insert(
            1,
            {
                "name": "wrist_context_adapter",
                "params": params(wrist_context_adapter),
                "lr": wrist_context_lr,
            },
        )
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )
    return optimizer


class SubtaskM2WTrainer(TrainerUtils):
    def __init__(self, cfg, model, dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.completed_steps = 0
        self.train_qwen_vl = bool(cfg.trainer.get("train_qwen_vl", False))
        self._pending_optimizer_state = None
        self._pending_lr_scheduler_state = None
        self._pending_rng_state = None
        self._resume_has_scheduler_state = False
        self.data_epoch_count = 0

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 42
        set_seed(seed)

        self._init_checkpointing()
        self._configure_trainable_modules()
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.dataloader,
        )
        self._restore_optimizer_scheduler_state()
        self._init_wandb()

    def _raw_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _configure_trainable_modules(self):
        raw = self._raw_model()
        for param in raw.qwen_vl_interface.parameters():
            param.requires_grad = self.train_qwen_vl
        for param in raw.visual_encoder.parameters():
            param.requires_grad = False
        raw.visual_encoder.eval()
        logger.info(
            "Configured trainable modules | "
            f"train_qwen_vl={self.train_qwen_vl} | V-JEPA2 frozen"
        )

    def _train_step(self, batch):
        with self.accelerator.accumulate(self.model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = self.model.forward(batch)
                total_loss = output["total_loss"]

            self.accelerator.backward(total_loss)
            did_step = self.accelerator.sync_gradients

            if did_step and self.config.trainer.get("gradient_clipping"):
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.trainer.gradient_clipping,
                )

            if did_step:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

        metrics = {
            key: value.detach().float().item() if torch.is_tensor(value) else float(value)
            for key, value in output.items()
        }
        return metrics, did_step

    def _set_data_epoch(self, epoch: int):
        """Propagate epoch to Accelerate wrappers, samplers, and LeRobot datasets."""
        dataloader = self.dataloader

        if callable(getattr(dataloader, "set_epoch", None)):
            dataloader.set_epoch(epoch)

        if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch)

        batch_sampler = getattr(dataloader, "batch_sampler", None)
        batch_sampler_inner = getattr(batch_sampler, "sampler", None)
        if callable(getattr(batch_sampler_inner, "set_epoch", None)):
            batch_sampler_inner.set_epoch(epoch)

        dataset = getattr(dataloader, "dataset", None)
        visited = set()
        while dataset is not None and id(dataset) not in visited:
            visited.add(id(dataset))
            if callable(getattr(dataset, "set_epoch", None)):
                dataset.set_epoch(epoch)
            dataset = getattr(dataset, "dataset", None)

    def _reset_data_iter(self, epoch_counter: int):
        epoch_counter += 1
        self._set_data_epoch(epoch_counter)
        return iter(self.dataloader), epoch_counter

    def train(self):
        logger.info(
            f"Starting SubtaskM2W training | max_steps={self.config.trainer.max_train_steps}"
        )
        grad_accum = int(self.config.trainer.get("gradient_accumulation_steps", 1))
        self.data_epoch_count = int(
            (self.completed_steps * grad_accum) // max(1, len(self.dataloader))
        )
        self._set_data_epoch(self.data_epoch_count)
        data_iter = iter(self.dataloader)
        progress = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_main_process,
        )
        self.model.train()

        while self.completed_steps < self.config.trainer.max_train_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter, self.data_epoch_count = self._reset_data_iter(self.data_epoch_count)
                batch = next(data_iter)

            metrics, did_step = self._train_step(batch)
            if not did_step:
                continue

            self.completed_steps += 1
            progress.update(1)
            step = self.completed_steps
            epoch = round((step * grad_accum) / max(1, len(self.dataloader)), 2)

            log_freq = self.config.trainer.get("logging_frequency", 20)
            if self.accelerator.is_main_process and (step == 1 or step % log_freq == 0):
                lr_by_name = {group["name"]: group["lr"] for group in self.optimizer.param_groups}
                vlm_cur_lr = lr_by_name.get("vlm_backbone", 0.0)
                logger.info(
                    f"step={step} | "
                    f"total={metrics.get('total_loss', 0):.4f} | "
                    f"action={metrics.get('action_loss', 0):.4f} | "
                    f"cot_weighted={metrics.get('cot_loss_weighted', 0):.4f} | "
                    f"jepa_weighted={metrics.get('jepa_loss_weighted', 0):.4f} | "
                    f"vlm_lr={vlm_cur_lr:.2e} | "
                    f"epoch={epoch:.2f}"
                )
                wandb_payload = {
                    # W&B groups metrics by the first slash-delimited prefix.
                    "loss/total": metrics.get("total_loss", 0),
                    "loss/action": metrics.get("action_loss", 0),
                    "loss/cot_weighted": metrics.get("cot_loss_weighted", 0),
                    "loss/jepa_weighted": metrics.get("jepa_loss_weighted", 0),
                    "loss_weight/lambda_cot": metrics.get("lambda_cot", 0),
                    "loss_weight/lambda_jepa": metrics.get("lambda_jepa", 0),
                    "train/epoch": epoch,
                }
                for name, lr in lr_by_name.items():
                    wandb_payload[f"lr/{name}"] = lr
                wandb.log(wandb_payload, step=step)

            save_interval = self.config.trainer.get("save_interval", 5000)
            if step > 0 and step % save_interval == 0:
                self._save_checkpoint(step)

        if self.completed_steps > 0:
            self._save_checkpoint(self.completed_steps)
        progress.close()
        logger.info("Training complete.")

    def _save_checkpoint(self, step: int):
        step_path = self._step_checkpoint_path(step)
        latest_path = self._latest_checkpoint_path()
        tmp_path = f"{step_path}.tmp"
        optimizer_state = None
        rng_state = self._capture_rng_state()
        if self._uses_deepspeed_optimizer():
            self._save_deepspeed_rank_state(step, rng_state)
            optimizer_state = self._rank_sharded_state_ref(step, "optimizer")
            rng_state = self._rank_sharded_state_ref(step, "rng")
        elif self.accelerator.is_main_process:
            optimizer_state = self.optimizer.state_dict()

        # Avoid accelerator.unwrap_model here: it imports deepspeed for type
        # checks, which can fail on systems without CUDA_HOME even when this
        # trainer is using plain DDP.
        if self.accelerator.is_main_process:
            unwrapped = self._raw_model()
            torch.save(
                {
                    "model_state_dict": unwrapped.state_dict(),
                    "jepa_predictor_state_dict": unwrapped.jepa_predictor.state_dict(),
                    "step": step,
                    "completed_steps": step,
                    "optimizer_state_dict": optimizer_state,
                    "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                    "rng_state": rng_state,
                },
                tmp_path,
            )
            os.replace(tmp_path, step_path)
            latest_tmp_path = f"{latest_path}.tmp"
            try:
                if os.path.exists(latest_tmp_path):
                    os.remove(latest_tmp_path)
                os.link(step_path, latest_tmp_path)
            except OSError:
                shutil.copy2(step_path, latest_tmp_path)
            os.replace(latest_tmp_path, latest_path)
            config_save_path = os.path.join(self.config.output_dir, "config.yaml")
            if not os.path.exists(config_save_path):
                OmegaConf.save(self.config, config_save_path)
            logger.info(f"Saved checkpoint: {step_path}")
            logger.info(f"Updated latest checkpoint: {latest_path}")
        self.accelerator.wait_for_everyone()

    def _rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    def _world_size(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    def _uses_deepspeed_optimizer(self) -> bool:
        optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
        optimizer_type = type(optimizer)
        return optimizer_type.__module__.startswith("deepspeed.") or optimizer_type.__name__.startswith("DeepSpeed")

    def _rank_shard_name(self, step: int, kind: str, rank: int) -> str:
        return f"steps_{step}_{kind}_rank{rank}.pt"

    def _rank_sharded_state_ref(self, step: int, kind: str) -> dict:
        return {
            "_format": "rank_sharded",
            "kind": kind,
            "world_size": self._world_size(),
            "rank_shard_pattern": f"steps_{step}_{kind}_rank{{rank}}.pt",
        }

    def _save_deepspeed_rank_state(self, step: int, rng_state: dict) -> None:
        rank = self._rank()
        optimizer_path = os.path.join(self.checkpoint_dir, self._rank_shard_name(step, "optimizer", rank))
        rng_path = os.path.join(self.checkpoint_dir, self._rank_shard_name(step, "rng", rank))
        torch.save(self.optimizer.state_dict(), f"{optimizer_path}.tmp")
        os.replace(f"{optimizer_path}.tmp", optimizer_path)
        torch.save(
            {
                "rank": rank,
                "world_size": self._world_size(),
                "step": step,
                "rng_state": rng_state,
            },
            f"{rng_path}.tmp",
        )
        os.replace(f"{rng_path}.tmp", rng_path)

    def _init_wandb(self):
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="subtask-m2w-train",
            )

    def _init_checkpointing(self):
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        pretrained = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)

        if is_resume:
            ckpt, self.completed_steps = self._get_resume_checkpoint()
            if ckpt:
                self.model = self.load_pretrained_backbones(self.model, ckpt)
                self._capture_training_state(ckpt)
                logger.info(f"Resumed from {ckpt}, step={self.completed_steps}")
                return

        if pretrained:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained, reload_modules)
            # Pretrained initialization is deliberately weight-only. Optimizer,
            # scheduler, RNG, and global-step state belong exclusively to resume.
            self.completed_steps = 0
            self._pending_checkpoint_path = None
            self._pending_optimizer_state = None
            self._pending_lr_scheduler_state = None
            self._pending_rng_state = None
            self._resume_has_scheduler_state = False
            logger.info(f"Loaded pretrained model weights only: {pretrained}, step=0")
        else:
            self.completed_steps = 0

    def _latest_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "latest_pytorch_model.pt")

    def _step_checkpoint_path(self, step: int) -> str:
        return os.path.join(self.checkpoint_dir, f"steps_{step}_pytorch_model.pt")

    def _get_resume_checkpoint(self):
        latest = self._latest_checkpoint_path()
        if not os.path.exists(latest):
            logger.info(f"No latest checkpoint found at {latest}")
            return None, 0
        try:
            checkpoint = self._load_checkpoint_file(latest)
        except Exception as exc:
            raise RuntimeError(f"loading latest checkpoint metadata failed: {exc}")
        completed_steps = 0
        if isinstance(checkpoint, dict):
            completed_steps = int(checkpoint.get("completed_steps", checkpoint.get("step", 0)))
        logger.info(f"Latest checkpoint found: {latest}, step={completed_steps}")
        return latest, completed_steps

    def _capture_rng_state(self):
        numpy_state = np.random.get_state()
        return {
            "python": random.getstate(),
            "numpy": {
                "name": numpy_state[0],
                "keys": numpy_state[1].tolist(),
                "pos": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _load_checkpoint_file(path: str):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _capture_training_state(self, checkpoint_path: str):
        self._pending_checkpoint_path = checkpoint_path
        try:
            checkpoint = self._load_checkpoint_file(checkpoint_path)
        except Exception as e:
            raise RuntimeError(f"loading checkpoint metadata failed: {e}")
        if not isinstance(checkpoint, dict):
            return
        self.completed_steps = int(checkpoint.get("completed_steps", checkpoint.get("step", self.completed_steps)))
        if bool(self.config.trainer.get("resume_optimizer_state", True)):
            self._pending_optimizer_state = checkpoint.get("optimizer_state_dict")
            self._pending_lr_scheduler_state = checkpoint.get("lr_scheduler_state_dict")
            self._resume_has_scheduler_state = self._pending_lr_scheduler_state is not None
            self._pending_rng_state = checkpoint.get("rng_state")

    def _restore_optimizer_scheduler_state(self):
        optimizer_state = self._resolve_pending_optimizer_state()
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
            self._move_optimizer_state_to_device()
            logger.info("Restored optimizer state from checkpoint.")
        else:
            logger.info("No optimizer state found in checkpoint; continuing with a fresh optimizer.")

        if self._pending_lr_scheduler_state is not None:
            self.lr_scheduler.load_state_dict(self._pending_lr_scheduler_state)
            logger.info("Restored LR scheduler state from checkpoint.")
        else:
            self._adjust_lr_scheduler_for_resume()

        self._restore_rng_state()

    def _resolve_pending_optimizer_state(self):
        state = self._pending_optimizer_state
        if state is None:
            return None

        if self._is_rank_sharded_state_ref(state):
            if not self._uses_deepspeed_optimizer():
                logger.warning("Skipping rank-sharded optimizer state because current optimizer is not DeepSpeed.")
                return None
            return self._load_rank_sharded_optimizer_state(state)

        if self._uses_deepspeed_optimizer():
            if self._is_deepspeed_optimizer_state_list(state):
                return state
            if self._looks_like_single_deepspeed_zero_state(state):
                logger.warning(
                    "Skipping optimizer restore: checkpoint contains only one DeepSpeed ZeRO optimizer "
                    "partition. A new checkpoint saved by this script will store per-rank optimizer shards."
                )
                return None
            logger.warning("Skipping optimizer restore: checkpoint optimizer state is not DeepSpeed rank-sharded.")
            return None

        return state

    @staticmethod
    def _is_rank_sharded_state_ref(state) -> bool:
        return isinstance(state, dict) and state.get("_format") == "rank_sharded"

    @staticmethod
    def _is_deepspeed_optimizer_state_list(state) -> bool:
        if isinstance(state, (list, tuple)):
            return True
        return isinstance(state, dict) and bool(state) and all(isinstance(key, int) for key in state.keys())

    @staticmethod
    def _looks_like_single_deepspeed_zero_state(state) -> bool:
        return (
            isinstance(state, dict)
            and "base_optimizer_state" in state
            and "single_partition_of_fp32_groups" in state
            and "zero_stage" in state
        )

    def _rank_shard_path(self, ref: dict, rank: int | None = None) -> str:
        if rank is None:
            rank = self._rank()
        checkpoint_path = getattr(self, "_pending_checkpoint_path", None)
        checkpoint_dir = os.path.dirname(checkpoint_path) if checkpoint_path else self.checkpoint_dir
        pattern = ref["rank_shard_pattern"]
        return os.path.join(checkpoint_dir, pattern.format(rank=rank))

    def _load_rank_sharded_optimizer_state(self, ref: dict):
        expected_world_size = int(ref.get("world_size", self._world_size()))
        world_size = self._world_size()
        if expected_world_size != world_size:
            logger.warning(
                "Skipping optimizer restore: checkpoint optimizer shards were saved with "
                f"world_size={expected_world_size}, current world_size={world_size}."
            )
            return None
        rank = self._rank()
        shard_path = self._rank_shard_path(ref, rank)
        if not os.path.exists(shard_path):
            logger.warning(f"Skipping optimizer restore: missing optimizer shard {shard_path}")
            return None
        return {rank: self._load_checkpoint_file(shard_path)}

    def _move_optimizer_state_to_device(self):
        optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
        if not hasattr(optimizer, "state") and hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer
        if not hasattr(optimizer, "state"):
            return
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(self.accelerator.device)

    def _adjust_lr_scheduler_for_resume(self):
        for _ in range(self.completed_steps):
            self.lr_scheduler.step()

    def _restore_rng_state(self):
        rng_state = getattr(self, "_pending_rng_state", None)
        if not rng_state:
            return
        if self._is_rank_sharded_state_ref(rng_state):
            shard_path = self._rank_shard_path(rng_state)
            if not os.path.exists(shard_path):
                logger.warning(f"Skipping RNG restore: missing RNG shard {shard_path}")
                return
            rng_shard = self._load_checkpoint_file(shard_path)
            rng_state = rng_shard.get("rng_state") if isinstance(rng_shard, dict) else None
            if not rng_state:
                logger.warning(f"Skipping RNG restore: invalid RNG shard {shard_path}")
                return
        try:
            random.setstate(rng_state["python"])
            numpy_state = rng_state["numpy"]
            if isinstance(numpy_state, dict):
                numpy_state = (
                    numpy_state["name"],
                    np.asarray(numpy_state["keys"], dtype=np.uint32),
                    int(numpy_state["pos"]),
                    int(numpy_state["has_gauss"]),
                    float(numpy_state["cached_gaussian"]),
                )
            np.random.set_state(numpy_state)
            torch.set_rng_state(rng_state["torch"])
            cuda_state = rng_state.get("cuda")
            if cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_state)
            logger.info("Restored RNG state from checkpoint.")
        except Exception as exc:
            logger.warning(f"Failed to restore RNG state from checkpoint: {exc}")


def main(cfg):
    bind_local_cuda_device()
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(cfg.trainer.get("find_unused_parameters", False))
    )
    deepspeed_plugin = (
        DeepSpeedPlugin()
        if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true"
        else None
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.trainer.get("gradient_accumulation_steps", 1)),
        kwargs_handlers=[ddp_kwargs],
        deepspeed_plugin=deepspeed_plugin,
    )
    output_dir = setup_directories(cfg)
    save_run_config(cfg, output_dir)
    if dist.is_initialized():
        dist.barrier()
    model = build_model(cfg)
    dataloader = prepare_data(cfg, accelerator)
    optimizer = build_optimizer(model, cfg)
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    trainer = SubtaskM2WTrainer(
        cfg=cfg,
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    trainer.prepare_training()
    trainer.train()

    logger.info("Done!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    main(cfg)
