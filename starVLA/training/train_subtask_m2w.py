"""
Trainer for QwenSubtaskM2W.

Default:
  - Single-stage training: Qwen-VL, main-to-wrist adapter, and action head are
    trainable from step 0.
  - Frozen wrist V-JEPA2.1 is used only as a latent teacher.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import (
    DistributedDataParallelKwargs,
    GradientAccumulationPlugin,
    set_seed,
)
from omegaconf import OmegaConf
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


def build_model(cfg):
    logger.info(f"Building QwenSubtaskM2W from `{cfg.framework.qwenvl.base_vlm}`")
    return build_framework(cfg)


def prepare_data(cfg, accelerator, output_dir):
    logger.info(f"Creating dataset with mix `{cfg.datasets.vla_data.data_mix}`")
    dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return dataloader


def build_optimizer(model, cfg):
    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 2.5e-5)
    adapter_lr = lr_cfg.get("adapter", lr_cfg.get("prompt", base_lr * 4))
    action_lr = lr_cfg.get("action_model", base_lr)
    vlm_lr = lr_cfg.get("vlm", base_lr * 0.2)
    train_qwen_vl = bool(cfg.trainer.get("train_qwen_vl", False))
    vlm_initial_lr = vlm_lr if train_qwen_vl else 0.0

    def trainable_params(module):
        if module is None:
            return []
        return [parameter for parameter in module.parameters() if parameter.requires_grad]

    param_groups = []

    m2w_params = trainable_params(model.m2w_adapter)
    if m2w_params:
        param_groups.append({"name": "m2w_adapter", "params": m2w_params, "lr": adapter_lr})

    action_params = trainable_params(model.action_model)
    if action_params:
        param_groups.append({"name": "action_heads", "params": action_params, "lr": action_lr})

    # V-JEPA is a frozen teacher and should not be handed to DeepSpeed ZeRO-2 as
    # a zero-lr/frozen optimizer group. Empty or all-frozen groups can create
    # empty ZeRO bit16 groups and fail during DeepSpeed optimizer initialization.
    if train_qwen_vl:
        for parameter in model.qwen_vl_interface.parameters():
            parameter.requires_grad = True
        qwen_params = trainable_params(model.qwen_vl_interface)
        if qwen_params:
            # Use the target VLM LR when constructing the scheduler. If this
            # group starts at 0, scheduler.base_lrs also becomes 0 and later
            # manual unfreezing is overwritten back to 0 at every scheduler step.
            param_groups.append({"name": "vlm_backbone", "params": qwen_params, "lr": vlm_initial_lr})

    if not param_groups:
        raise ValueError("No trainable parameter groups were found for QwenSubtaskM2W.")

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )
    return optimizer, vlm_lr


class SubtaskM2WTrainer(TrainerUtils):
    def __init__(self, cfg, model, dataloader, optimizer, vlm_lr, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.vlm_lr = vlm_lr
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.completed_steps = 0
        self.train_qwen_vl = bool(cfg.trainer.get("train_qwen_vl", False))
        self._pending_optimizer_state = None
        self._pending_lr_scheduler_state = None
        self._resume_has_scheduler_state = False

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
        self._config_save_path = self._save_config()
        self._init_wandb()

    def _raw_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _configure_trainable_modules(self):
        raw = self._raw_model()
        for param in raw.qwen_vl_interface.parameters():
            param.requires_grad = self.train_qwen_vl
        visual_encoder = getattr(raw, "visual_encoder", None)
        if visual_encoder is not None:
            for param in visual_encoder.parameters():
                param.requires_grad = False
            visual_encoder.eval()
        if self.train_qwen_vl:
            logger.info("[Single Stage] Qwen-VL trainable from step 0. V-JEPA2 frozen.")
        else:
            logger.info("[Single Stage] Qwen-VL frozen by config. V-JEPA2 frozen.")

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
                m2w_adapter = getattr(self._raw_model(), "m2w_adapter", None)
                if m2w_adapter is not None:
                    m2w_adapter.update_target_ema()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

        metrics = {key: value.item() if hasattr(value, "item") else float(value) for key, value in output.items()}
        return metrics, did_step

    def _set_data_epoch(self, epoch: int):
        """Propagate epoch to the sampler and LeRobot mixture dataset."""
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

    def _reset_data_iter(self, epoch: int):
        epoch += 1
        self._set_data_epoch(epoch)
        return iter(self.dataloader), epoch

    def train(self):
        logger.info(
            f"Starting SubtaskM2W training | train_qwen_vl={self.train_qwen_vl} | "
            f"max_steps={self.config.trainer.max_train_steps}"
        )
        grad_accum = int(self.config.trainer.get("gradient_accumulation_steps", 1))
        data_epoch = int((self.completed_steps * grad_accum) // max(len(self.dataloader), 1))
        self._set_data_epoch(data_epoch)
        data_iter = iter(self.dataloader)
        progress = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_main_process,
        )
        self.model.train()

        while self.completed_steps < self.config.trainer.max_train_steps:
            data_start = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter, data_epoch = self._reset_data_iter(data_epoch)
                batch = next(data_iter)
            data_time = time.perf_counter() - data_start

            model_start = time.perf_counter()
            metrics, did_step = self._train_step(batch)
            model_time = time.perf_counter() - model_start
            if not did_step:
                continue

            self.completed_steps += 1
            progress.update(1)
            step = self.completed_steps
            try:
                epoch = round((step * grad_accum) / len(self.dataloader), 2)
            except TypeError:
                epoch = 0.0
            metrics["data_time"] = data_time
            metrics["model_time"] = model_time
            metrics["epoch"] = epoch
            if self.accelerator.is_main_process:
                progress.set_postfix(
                    data_times=f"{data_time:.3f}",
                    model_times=f"{model_time:.3f}",
                )

            log_freq = self.config.trainer.get("logging_frequency", 20)
            if self.accelerator.is_main_process and (step == 1 or step % log_freq == 0):
                group_lrs = {
                    group.get("name", f"group_{idx}"): group["lr"]
                    for idx, group in enumerate(self.optimizer.param_groups)
                }
                vlm_cur_lr = group_lrs.get("vlm_backbone", 0.0)
                adapter_cur_lr = group_lrs.get("m2w_adapter", 0.0)
                wrist_metric = metrics.get("wrist_latent_loss", 0)
                logger.info(
                    f"step={step} | "
                    f"total={metrics.get('total_loss', 0):.4f} | "
                    f"action={metrics.get('action_loss', 0):.4f} | "
                    f"cot={metrics.get('cot_loss', 0):.4f} | "
                    f"wrist_latent={wrist_metric:.4f} | "
                    f"vlm_lr={vlm_cur_lr:.2e} | "
                    f"epoch={epoch:.2f} | "
                    f"data_time={data_time:.3f} | "
                    f"model_time={model_time:.3f}"
                )
                wandb.log(
                    {
                        "train/total_loss": metrics.get("total_loss", 0),
                        "train/action_loss": metrics.get("action_loss", 0),
                        "train/cot_loss": metrics.get("cot_loss", 0),
                        "train/wrist_latent_loss": wrist_metric,
                        "train/lr_adapter": adapter_cur_lr,
                        "train/lr_vlm": vlm_cur_lr,
                        "train/data_time": data_time,
                        "train/model_time": model_time,
                        "train/epoch": epoch,
                    },
                    step=step,
                )

            save_interval = self.config.trainer.get("save_interval", 5000)
            if self.accelerator.is_main_process and step > 0 and step % save_interval == 0:
                self._save_checkpoint(step)

        progress.close()
        logger.info("Training complete.")

    def _save_checkpoint(self, step: int):
        ckpt_dir = os.path.join(self.config.output_dir, "checkpoints")
        save_path = os.path.join(ckpt_dir, f"steps_{step}_pytorch_model.pt")
        # Avoid accelerator.unwrap_model here: it imports deepspeed for type
        # checks, which can fail on systems without CUDA_HOME even when this
        # trainer is using plain DDP.
        unwrapped = self._raw_model()
        checkpoint = {
            "model_state_dict": unwrapped.state_dict(),
            "step": step,
            "completed_steps": step,
            **self._optional_training_state(),
        }
        m2w_adapter = getattr(unwrapped, "m2w_adapter", None)
        if m2w_adapter is not None:
            checkpoint["m2w_adapter_state_dict"] = m2w_adapter.state_dict()
        torch.save(checkpoint, save_path)
        if not os.path.exists(getattr(self, "_config_save_path", "")):
            self._config_save_path = self._save_config()
        logger.info(f"Saved checkpoint: {save_path}")

    def _save_config(self):
        config_save_path = os.path.join(self.config.output_dir, "config.yaml")
        if self.accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            OmegaConf.save(self.config, config_save_path)
            logger.info(f"Saved run config: {config_save_path}")
        self.accelerator.wait_for_everyone()
        return config_save_path

    def _init_wandb(self):
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="subtask-m2w-train",
            )
            config_save_path = getattr(self, "_config_save_path", None)
            if config_save_path and os.path.exists(config_save_path):
                try:
                    wandb.save(
                        config_save_path,
                        base_path=self.config.output_dir,
                        policy="now",
                    )
                    logger.info("Uploaded config.yaml to W&B.")
                except Exception as exc:
                    logger.warning(f"Failed to upload config.yaml to W&B: {exc}")

    def _init_checkpointing(self):
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        pretrained = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)

        if is_resume:
            ckpt, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if ckpt:
                self.model = self.load_pretrained_backbones(self.model, ckpt)
                self._capture_training_state(ckpt)
                logger.info(f"Resumed from {ckpt}, step={self.completed_steps}")
                return

        if pretrained:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained, reload_modules)
            self._capture_training_state(pretrained)
            if self.completed_steps == 0:
                try:
                    self.completed_steps = int(
                        re.search(r"steps_(\d+)_pytorch_model\.pt", pretrained).group(1)
                    )
                except AttributeError:
                    self.completed_steps = 0
            logger.info(f"Loaded pretrained: {pretrained}, step={self.completed_steps}")
        else:
            self.completed_steps = 0

    def _optional_training_state(self):
        if not bool(self.config.trainer.get("save_optimizer_state", False)):
            return {}
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
        }

    def _capture_training_state(self, checkpoint_path: str):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"loading checkpoint metadata failed: {e}")
        if not isinstance(checkpoint, dict):
            return
        self.completed_steps = int(checkpoint.get("completed_steps", checkpoint.get("step", self.completed_steps)))
        if bool(self.config.trainer.get("resume_optimizer_state", True)):
            self._pending_optimizer_state = checkpoint.get("optimizer_state_dict")
            self._pending_lr_scheduler_state = checkpoint.get("lr_scheduler_state_dict")
            self._resume_has_scheduler_state = self._pending_lr_scheduler_state is not None

    def _restore_optimizer_scheduler_state(self):
        if self._pending_optimizer_state is not None:
            self.optimizer.load_state_dict(self._pending_optimizer_state)
            self._move_optimizer_state_to_device()
            logger.info("Restored optimizer state from checkpoint.")
        else:
            logger.info("No optimizer state found in checkpoint; continuing with a fresh optimizer.")

        if self._pending_lr_scheduler_state is not None:
            self.lr_scheduler.load_state_dict(self._pending_lr_scheduler_state)
            logger.info("Restored LR scheduler state from checkpoint.")
        else:
            self._adjust_lr_scheduler_for_resume()

    def _move_optimizer_state_to_device(self):
        optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(self.accelerator.device)

    def _adjust_lr_scheduler_for_resume(self):
        for _ in range(self.completed_steps):
            self.lr_scheduler.step()


def main(cfg):
    bind_local_cuda_device()
    use_deepspeed = os.environ.get("STARVLA_USE_DEEPSPEED", "0").lower() not in (
        "0",
        "false",
        "no",
    )
    gradient_accumulation_steps = int(cfg.trainer.get("gradient_accumulation_steps", 1))
    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=gradient_accumulation_steps,
        # DeepSpeed ZeRO-2 is incompatible with Accelerate's no_sync path.
        # Plain DDP can keep the usual no_sync behavior.
        sync_each_batch=use_deepspeed,
    )
    accelerator_kwargs = {"gradient_accumulation_plugin": gradient_accumulation_plugin}
    if use_deepspeed:
        accelerator_kwargs["deepspeed_plugin"] = DeepSpeedPlugin()
    else:
        accelerator_kwargs["kwargs_handlers"] = [
            DistributedDataParallelKwargs(
                find_unused_parameters=bool(cfg.trainer.get("find_unused_parameters", False))
            )
        ]
    accelerator = Accelerator(**accelerator_kwargs)
    output_dir = setup_directories(cfg)
    model = build_model(cfg)
    dataloader = prepare_data(cfg, accelerator, output_dir)
    optimizer, vlm_lr = build_optimizer(model, cfg)
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
        vlm_lr=vlm_lr,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    trainer.prepare_training()
    trainer.train()

    logger.info("Done!")
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
