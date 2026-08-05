from __future__ import annotations
from collections import deque
import json
import os
from pathlib import Path
import time
from typing import Dict, Optional, Sequence

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from deployment.model_server.tools.adaptive_ensemble import AdaptiveEnsembler


def read_mode_config(pretrained_checkpoint):
    """Read run config and dataset statistics without importing starVLA training code."""
    checkpoint_path = Path(pretrained_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
    if checkpoint_path.suffix != ".pt":
        raise ValueError(f"Expected a .pt checkpoint, got `{checkpoint_path}`.")

    run_dir = checkpoint_path.parents[1]
    config_yaml = run_dir / "config.yaml"
    dataset_statistics_json = run_dir / "dataset_statistics.json"
    if not config_yaml.exists():
        raise FileNotFoundError(f"Missing `config.yaml` for `{run_dir}`.")
    if not dataset_statistics_json.exists():
        raise FileNotFoundError(f"Missing `dataset_statistics.json` for `{run_dir}`.")

    try:
        from omegaconf import OmegaConf

        ocfg = OmegaConf.load(str(config_yaml))
        model_config = OmegaConf.to_container(ocfg, resolve=True)
    except Exception:
        import yaml

        with open(config_yaml, "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f)

    with open(dataset_statistics_json, "r", encoding="utf-8") as f:
        norm_stats = json.load(f)
    return model_config, norm_stats


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble=True,
        action_ensemble_horizon: Optional[int] = 3,  # different cross sim
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="0.0.0.0",
        port=10095,
    ) -> None:

        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon  # 0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        self.jepa_prediction_view = self.get_jepa_prediction_view(policy_ckpt_path)
        self.predictor_history_frames = self.get_wrist_history_frames(policy_ckpt_path)
        self.main_image_history = deque(maxlen=max(1, self.predictor_history_frames))
        self.wrist_image_history = deque(maxlen=max(1, self.predictor_history_frames))
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)
        self.policy_state_dim = self.get_state_dim(policy_ckpt_path=policy_ckpt_path)
        self.policy_uses_state = self.get_include_state(policy_ckpt_path=policy_ckpt_path)
        self._state_adapt_logged = False
        self._state_usage_logged = False
        self.view_ablation = os.environ.get("VIEW_ABLATION", "dual").strip().lower()
        self.debug_outputs = self._env_flag("PRINT_M2W_OUTPUTS") or self._env_flag("PRINT_GENERATED_COT")
        self.debug_actions = self._env_flag("PRINT_M2W_ACTIONS")
        self.debug_every = self._env_int("PRINT_M2W_EVERY", 1)
        self.debug_jsonl = os.environ.get("M2W_DEBUG_JSONL", "").strip() or None
        if self.debug_jsonl:
            Path(self.debug_jsonl).parent.mkdir(parents=True, exist_ok=True)
        self.debug_latency = self._env_flag("PRINT_M2W_LATENCY")
        self.latency_jsonl = os.environ.get("M2W_LATENCY_JSONL", "").strip() or None
        if self.latency_jsonl:
            Path(self.latency_jsonl).parent.mkdir(parents=True, exist_ok=True)
        self.latency_records: list[dict] = []
        self.episode_latency_records: list[dict] = []

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        self.main_image_history.clear()
        self.wrist_image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.episode_latency_records.clear()

    def step(self, example: dict, step: int = 0, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Perform one step of inference
        :param image: Input image in the format (H, W, 3), type uint8
        :param task_description: Task description text
        :return: (raw action, processed action)
        """

        task_description = example.get("lang", None)
        images = example["image"]  # list of images for history

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        images = [self._resize_image(image) for image in images]
        images = self._apply_view_ablation(images)
        example["image"] = images
        self._adapt_state_for_policy(example)
        if self.jepa_prediction_view == "main" and self.predictor_history_frames > 1 and images:
            self.main_image_history.append(images[0])
            main_clip = self._build_history_clip(
                list(self.main_image_history),
                self.predictor_history_frames,
            )
            if main_clip:
                example["main_views"] = [main_clip]
        elif self.jepa_prediction_view == "wrist" and self.predictor_history_frames > 1 and len(images) >= 2:
            self.wrist_image_history.append(images[1])
            wrist_clip = self._build_history_clip(
                list(self.wrist_image_history),
                self.predictor_history_frames,
            )
            if wrist_clip:
                example["wrist_views"] = [wrist_clip]
        vla_input = {
            "examples": [example],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
            "_eval_metadata": {
                "task_description": task_description,
                "step": int(step),
                "chunk": int(step // self.action_chunk_size),
            },
        }

        action_chunk_size = self.action_chunk_size
        if step % action_chunk_size == 0:
            latency_start = time.perf_counter()
            response = self.client.predict_action(vla_input)
            latency_ms = (time.perf_counter() - latency_start) * 1000.0
            if not response.get("ok", True):
                error = response.get("error", {})
                message = error.get("message", str(error))
                raise RuntimeError(f"Policy server inference failed: {message}")
            try:
                data = response["data"]
                normalized_actions = data["normalized_actions"]  # B, chunk, D
            except KeyError:
                print(f"Response data: {response}")
                data = response.get("data", {})
                raise KeyError(f"Key 'normalized_actions' not found in response data: {data.keys()}")

            generated_cot = data.get("generated_cot")
            normalized_actions = np.asarray(normalized_actions[0], dtype=np.float32)
            self.raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats
            )
            self._record_chunk_latency(step, task_description, latency_ms)
            self._maybe_log_prediction(
                step,
                task_description,
                generated_cot,
                normalized_actions,
                self.raw_actions,
                latency_ms=latency_ms,
            )

        raw_actions = self.raw_actions[step % action_chunk_size][None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        return {"raw_action": raw_action}

    def _apply_view_ablation(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if self.view_ablation in {"", "dual", "both", "none"} or len(images) < 2:
            return images

        main = images[0]
        wrist = images[1]
        if self.view_ablation in {"main", "main_only", "primary", "primary_only"}:
            return [main, main]
        if self.view_ablation in {"wrist", "wrist_only"}:
            return [wrist, wrist]
        if self.view_ablation in {"main_single", "primary_single"}:
            return [main]
        if self.view_ablation == "wrist_single":
            return [wrist]
        if self.view_ablation == "swap":
            return [wrist, main]
        raise ValueError(
            "Unsupported VIEW_ABLATION value "
            f"`{self.view_ablation}`. Use dual, main_only, wrist_only, "
            "main_single, wrist_single, or swap."
        )

    @staticmethod
    def _build_history_clip(history: Sequence[np.ndarray], target_frames: int) -> list[np.ndarray]:
        """Return a fixed-length clip aligned with training-time boundary padding."""
        frames = list(history)
        target_frames = max(1, int(target_frames))
        if not frames:
            return []
        if len(frames) >= target_frames:
            return frames[-target_frames:]
        return [frames[0]] * (target_frames - len(frames)) + frames

    def _adapt_state_for_policy(self, example: dict) -> None:
        if not self.policy_uses_state:
            if "state" in example:
                example.pop("state", None)
                if not self._state_usage_logged:
                    print(
                        "[LIBERO eval] omitted state because checkpoint config does not set include_state=true",
                        flush=True,
                    )
                    self._state_usage_logged = True
            return

        if self.policy_state_dim <= 0 or "state" not in example:
            return

        state = np.asarray(example["state"])
        if state.shape[-1] == self.policy_state_dim:
            return

        original_dim = state.shape[-1]
        if self.policy_state_dim == 7 and original_dim == 8:
            # StarVLA maps index 6 as state.pad; 7D state keeps the final gripper channel.
            state = np.concatenate([state[..., :6], state[..., 7:8]], axis=-1)
        elif self.policy_state_dim == 8 and original_dim == 7:
            # Reinsert the placeholder to recover [xyz, rpy, pad, gripper].
            pad = np.zeros_like(state[..., :1])
            state = np.concatenate([state[..., :6], pad, state[..., 6:7]], axis=-1)
        elif original_dim > self.policy_state_dim:
            state = state[..., : self.policy_state_dim]
        else:
            pad_shape = list(state.shape)
            pad_shape[-1] = self.policy_state_dim - original_dim
            pad = np.zeros(pad_shape, dtype=state.dtype)
            state = np.concatenate([state, pad], axis=-1)

        example["state"] = state.astype(np.float32, copy=False)
        if not self._state_adapt_logged:
            print(
                f"[LIBERO eval] adapted state dim {original_dim} -> {self.policy_state_dim}",
                flush=True,
            )
            self._state_adapt_logged = True

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)  # read config and norm_stats
        return model_config["framework"]["action_model"]["action_horizon"]

    @staticmethod
    def get_state_dim(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)
        return int(model_config["framework"]["action_model"].get("state_dim", 0) or 0)

    @staticmethod
    def get_include_state(policy_ckpt_path) -> bool:
        model_config, _ = read_mode_config(policy_ckpt_path)
        vla_data_cfg = model_config.get("datasets", {}).get("vla_data", {})
        value = vla_data_cfg.get("include_state", False)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return bool(value)

    @staticmethod
    def get_wrist_history_frames(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)
        data_cfg = model_config.get("datasets", {}).get("vla_data", {})
        return max(1, int(data_cfg.get("wrist_history_frames", 1)))

    @staticmethod
    def get_jepa_prediction_view(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)
        framework_cfg = model_config.get("framework", {})
        jepa_cfg = framework_cfg.get("jepa_predictor", {})
        data_cfg = model_config.get("datasets", {}).get("vla_data", {})
        view = str(jepa_cfg.get("prediction_view", data_cfg.get("jepa_prediction_view", "wrist")))
        view = view.strip().lower()
        if view not in {"wrist", "main"}:
            raise ValueError(
                f"Unsupported JEPA prediction view `{view}` in checkpoint config. "
                "Expected `wrist` or `main`."
            )
        print(f"[LIBERO eval] JEPA prediction view: {view}", flush=True)
        return view

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def _maybe_log_prediction(
        self,
        step: int,
        task_description: Optional[str],
        generated_cot,
        normalized_actions: np.ndarray,
        raw_actions: np.ndarray,
        latency_ms: Optional[float] = None,
    ) -> None:
        chunk_index = step // self.action_chunk_size
        if chunk_index % self.debug_every != 0:
            return

        cot_text = self._first_batch_item(generated_cot)
        should_print = self.debug_outputs or self.debug_actions
        if should_print:
            print(
                f"[M2W eval] step={step} chunk={chunk_index} "
                f"task={task_description!r}",
                flush=True,
            )
        if self.debug_outputs:
            print(f"[M2W eval] generated_cot: {cot_text}", flush=True)
        if self.debug_actions:
            print(f"[M2W eval] normalized_action[0]: {self._round_list(normalized_actions[0])}", flush=True)
            print(f"[M2W eval] raw_action[0]: {self._round_list(raw_actions[0])}", flush=True)

        if self.debug_jsonl:
            record = {
                "step": int(step),
                "chunk": int(chunk_index),
                "task": task_description,
                "generated_cot": cot_text,
                "normalized_action_first": self._round_list(normalized_actions[0]),
                "raw_action_first": self._round_list(raw_actions[0]),
            }
            if latency_ms is not None:
                record["chunk_latency_ms"] = round(float(latency_ms), 3)
            with open(self.debug_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _record_chunk_latency(self, step: int, task_description: Optional[str], latency_ms: float) -> None:
        chunk_index = step // self.action_chunk_size
        record = {
            "time_unix": time.time(),
            "step": int(step),
            "chunk": int(chunk_index),
            "task": task_description,
            "action_chunk_size": int(self.action_chunk_size),
            "latency_ms": float(latency_ms),
        }
        self.latency_records.append(record)
        self.episode_latency_records.append(record)

        if self.debug_latency:
            episode_summary = self.get_episode_latency_summary()
            total_summary = self.get_latency_summary()
            print(
                "[M2W latency] "
                f"step={step} chunk={chunk_index} "
                f"latency_ms={latency_ms:.2f} "
                f"episode_avg_ms={episode_summary.get('avg_ms', 0.0):.2f} "
                f"total_avg_ms={total_summary.get('avg_ms', 0.0):.2f}",
                flush=True,
            )

        if self.latency_jsonl:
            with open(self.latency_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def latency_record_count(self) -> int:
        return len(self.latency_records)

    def get_episode_latency_summary(self) -> dict:
        return self.get_latency_summary(records=self.episode_latency_records)

    def get_latency_summary(
        self,
        start_index: int = 0,
        end_index: Optional[int] = None,
        records: Optional[Sequence[dict]] = None,
    ) -> dict:
        selected = list(records) if records is not None else self.latency_records[start_index:end_index]
        if not selected:
            return {"count": 0}
        values = np.asarray([float(item["latency_ms"]) for item in selected], dtype=np.float64)
        return {
            "count": int(values.size),
            "avg_ms": float(values.mean()),
            "median_ms": float(np.median(values)),
            "p90_ms": float(np.percentile(values, 90)),
            "p95_ms": float(np.percentile(values, 95)),
            "min_ms": float(values.min()),
            "max_ms": float(values.max()),
        }

    @staticmethod
    def format_latency_summary(label: str, summary: dict) -> str:
        if not summary or int(summary.get("count", 0)) == 0:
            return f"{label}: no action chunks recorded."
        return (
            f"{label}: Average inference latency per action chunk = {summary['avg_ms']:.2f} ms "
            f"(median={summary['median_ms']:.2f}, p90={summary['p90_ms']:.2f}, "
            f"p95={summary['p95_ms']:.2f}, min={summary['min_ms']:.2f}, "
            f"max={summary['max_ms']:.2f}, chunks={summary['count']})."
        )

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None or value.strip() == "":
            return default
        try:
            return max(1, int(value))
        except ValueError:
            return default

    @staticmethod
    def _first_batch_item(value):
        if isinstance(value, (list, tuple)) and value:
            return value[0]
        return value

    @staticmethod
    def _round_list(value: np.ndarray, ndigits: int = 4) -> list[float]:
        return np.round(np.asarray(value, dtype=np.float32), ndigits).tolist()

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
