from collections import deque
from pathlib import Path
from typing import Dict, Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.tools import read_mode_config

try:
    from deployment.model_server.tools.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble=False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="127.0.0.1",
        port=5694,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
    ) -> None:

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode} ***"
        )
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.normalization_mode = normalization_mode

        # Action mode: "abs", "delta", or "rel"
        self.action_mode = action_mode
        # State tracking for delta/rel modes
        self.initial_state = None  # s_0 for rel mode
        self.prev_action = None  # last absolute action for delta mode

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        (
            self.wrist_history_frames,
            self.wrist_temporal_stride,
        ) = self.get_wrist_history_config(policy_ckpt_path)
        self.wrist_history_span = (
            (self.wrist_history_frames - 1) * self.wrist_temporal_stride + 1
        )
        self.left_wrist_history = deque(maxlen=self.wrist_history_span)
        self.right_wrist_history = deque(maxlen=self.wrist_history_span)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_chunk_size = None
        self.state_norm_stats = None
        self.raw_actions = None
        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        config_action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)

        server_meta = self.client.get_server_metadata()
        self.action_chunk_size = server_meta.get("action_chunk_size") or config_action_chunk_size
        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode}, "
            f"wrist_history_frames: {self.wrist_history_frames}, "
            f"wrist_temporal_stride: {self.wrist_temporal_stride}, "
            f"wrist_history_span: {self.wrist_history_span}, "
            f"server_meta: {server_meta} ***"
        )

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        self.left_wrist_history.clear()
        self.right_wrist_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None
        # Reset state tracking for delta/rel modes
        self.initial_state = None
        self.prev_action = None

    def step(
        self,
        example: dict,
        step: int = 0,
    ) -> np.ndarray:
        state = example.get("state", None)
        # if state is not None:
        #     state = self.normalize_state(state, self.state_norm_stats)
        #     state = state[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]]
        #     example["state"] = state.reshape(1, -1)

        # Store initial state for delta/rel modes
        if self.action_mode in ["delta", "rel"] and self.initial_state is None:
            if state is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state to be provided in example")
            self.initial_state = np.array(state).copy()

        task_description = example.get("lang", None)
        images = example["image"]

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)
                # Re-store initial state after reset if in delta/rel mode
                if self.action_mode in ["delta", "rel"] and state is not None:
                    self.initial_state = np.array(state).copy()

        images = [self._resize_image(image) for image in images]
        if len(images) >= 3:
            self.left_wrist_history.append(images[1])
            self.right_wrist_history.append(images[2])
            example["wrist_views"] = [
                self._history_clip(self.left_wrist_history),
                self._history_clip(self.right_wrist_history),
            ]
        example["image"] = images
        example_copy = example.copy()
        example_copy.pop("state")
        vla_input = {
            "examples": [example_copy],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        vla_input["unnorm_key"] = self.unnorm_key

        action_chunk_size = self.action_chunk_size

        if action_chunk_size is None or step % action_chunk_size == 0 or self.raw_actions is None:
            # === TRAIN/TEST CONSISTENCY: keep the observation below aligned with training ===
            # Embodied policies degrade SILENTLY (no error) when the eval-time observation
            # differs from what the model saw during TRAINING. Verify these match the
            # training config used for this checkpoint:
            #   - state       : whether proprioceptive state is included (and its dim/order/normalization))
            #   - image size  : resize / crop resolution (e.g. 224x224)
            #   - image count : how many camera views are fed
            #   - image order : the ordering of those camera views
            #   - action normalization: unnorm_key must match the training dataset stats
            # ==============================================================================
            response = self.client.predict_action(vla_input)
            if not response.get("ok", True):
                error = response.get("error", {})
                message = error.get("message", str(error))
                raise RuntimeError(f"Policy server inference failed: {message}")
            data = response.get("data", {})
            if "actions" in data:
                # Some deployment servers may return already un-normalized actions.
                raw_actions = np.asarray(data["actions"][0], dtype=np.float32)  # (chunk, D)
            elif "normalized_actions" in data:
                normalized_actions = np.asarray(data["normalized_actions"][0], dtype=np.float32)
                raw_actions = self.unnormalize_actions(
                    normalized_actions=normalized_actions,
                    action_norm_stats=self.action_norm_stats,
                )
            else:
                raise KeyError(f"Expected `actions` or `normalized_actions` in response data, got: {data.keys()}")
            if self.action_chunk_size is None:
                self.action_chunk_size = len(raw_actions)
                action_chunk_size = self.action_chunk_size
                print(f"*** inferred action_chunk_size from first response: {self.action_chunk_size} ***")

            # Convert delta/rel to absolute actions
            if self.action_mode == "delta":
                self.raw_actions = self._delta_to_absolute(raw_actions, state)
            elif self.action_mode == "rel":
                self.raw_actions = self._rel_to_absolute(raw_actions)
            else:
                self.raw_actions = raw_actions

        action_idx = step % action_chunk_size
        if action_idx >= len(self.raw_actions):
            raise RuntimeError(
                "Action chunk length mismatch: "
                f"step={step}, action_idx={action_idx}, "
                f"configured_chunk_size={action_chunk_size}, "
                f"returned_chunk_length={len(self.raw_actions)}"
            )

        current_action = self.raw_actions[action_idx]

        # Update prev_action for delta mode (for cross-chunk continuity)
        if self.action_mode == "delta":
            self.prev_action = current_action.copy()

        current_action = current_action[[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]
        return current_action

    def _delta_to_absolute(self, delta_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """Convert delta actions to absolute actions."""
        abs_actions = np.zeros_like(delta_actions)
        base = self.prev_action if self.prev_action is not None else self.initial_state
        for i in range(len(delta_actions)):
            abs_actions[i] = delta_actions[i] + base
            base = abs_actions[i]
        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        """Convert relative actions to absolute actions."""
        return rel_actions + self.initial_state

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def _history_clip(self, history: deque) -> list[np.ndarray]:
        if not history:
            raise ValueError("Wrist history is empty.")
        clip = list(history)
        if len(clip) < self.wrist_history_span:
            clip = [clip[0]] * (self.wrist_history_span - len(clip)) + clip
        window = clip[-self.wrist_history_span :]
        sampled = window[:: self.wrist_temporal_stride]
        if len(sampled) != self.wrist_history_frames:
            raise RuntimeError(
                "Invalid wrist history sampling: "
                f"span={self.wrist_history_span}, stride={self.wrist_temporal_stride}, "
                f"expected={self.wrist_history_frames}, got={len(sampled)}"
            )
        return [frame.copy() for frame in sampled]

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high = np.asarray(action_norm_stats["max"], dtype=np.float32)
        action_low = np.asarray(action_norm_stats["min"], dtype=np.float32)
        normalized_actions = np.clip(normalized_actions, -1, 1)
        return np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        _, norm_stats = read_mode_config(Path(policy_ckpt_path))
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path) -> int:
        model_config, _ = read_mode_config(Path(policy_ckpt_path))
        return int(model_config["framework"]["action_model"]["future_action_window_size"]) + 1

    @staticmethod
    def get_wrist_history_config(policy_ckpt_path) -> tuple[int, int]:
        model_config, _ = read_mode_config(Path(policy_ckpt_path))
        vla_data = model_config.get("datasets", {}).get("vla_data", {})
        frames = max(1, int(vla_data.get("wrist_history_frames", 1) or 1))

        configured_stride = vla_data.get("wrist_temporal_stride")
        if configured_stride is None:
            dataset_py = str(vla_data.get("dataset_py", ""))
            # Existing Robotwin M2W checkpoints predate the explicit config
            # field, while their dataloader hard-codes temporal stride 2.
            if frames == 1:
                stride = 1
            elif "subtask_m2w_robotwin" in dataset_py:
                stride = 2
            else:
                raise ValueError(
                    "Cannot infer wrist_temporal_stride for a multi-frame checkpoint: "
                    f"dataset_py={dataset_py!r}, wrist_history_frames={frames}. "
                    "Set datasets.vla_data.wrist_temporal_stride explicitly."
                )
        else:
            stride = max(1, int(configured_stride))
        return frames, stride

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, choose one of: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))
        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` is not in the available dataset statistics: {norm_stats.keys()}"
        )
        return unnorm_key


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    action_mode = usr_args.get("action_mode", "abs")
    normalization_mode = usr_args.get(
        "action_normalization_mode",
        usr_args.get("normalization_mode", "min_max"),
    )

    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        action_mode=action_mode,
        normalization_mode=normalization_mode,
    )


def reset_model(model):
    model.reset(task_description="")


def eval(TASK_ENV, model, observation):
    # Get instruction
    instruction = TASK_ENV.get_instruction()

    # Prepare images
    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]

    # Order: [head, left, right] to match training order
    images = [head_img, left_img, right_img]

    state = observation["joint_action"]["vector"]
    example = {
        "lang": str(instruction),
        "image": images,
        "state": state,  # Required for delta/rel action modes
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)

    # Execute action
    TASK_ENV.take_action(action)
