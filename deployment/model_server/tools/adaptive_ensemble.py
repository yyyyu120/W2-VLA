"""Temporal action ensembling used by simulator policy clients."""

from collections import deque

import numpy as np


class AdaptiveEnsembler:
    def __init__(self, pred_action_horizon, adaptive_ensemble_alpha=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            current_predictions = np.stack(self.action_history)
        else:
            current_predictions = np.stack(
                [
                    predicted_actions[index]
                    for index, predicted_actions in zip(
                        range(num_actions - 1, -1, -1),
                        self.action_history,
                    )
                ]
            )

        reference = current_predictions[num_actions - 1]
        dot_product = np.sum(current_predictions * reference, axis=1)
        prediction_norm = np.linalg.norm(current_predictions, axis=1)
        reference_norm = np.linalg.norm(reference)
        cosine_similarity = dot_product / (
            prediction_norm * reference_norm + 1e-7
        )

        weights = np.exp(self.adaptive_ensemble_alpha * cosine_similarity)
        weights = weights / weights.sum()
        return np.sum(weights[:, None] * current_predictions, axis=0)
