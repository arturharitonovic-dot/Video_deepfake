import numpy as np
from collections import deque


class TemporalAnalyzer:
    def __init__(self, window_size: int = 30, variance_threshold: float = 0.005):
        self.window_size = window_size
        self.variance_threshold = variance_threshold
        self.history = deque(maxlen=window_size)

        self.alpha = 0.25
        self.smoothed_prob = 0.0

    def update(self, current_prob: float) -> dict:
        self.history.append(current_prob)

        if len(self.history) == 1:
            self.smoothed_prob = current_prob
        else:
            self.smoothed_prob = self.alpha * current_prob + (1 - self.alpha) * self.smoothed_prob

        history_list = list(self.history)
        if len(history_list) > 1:
            diffs = [abs(history_list[i] - history_list[i - 1]) for i in range(1, len(history_list))]
            variance = float(np.var(diffs))
        else:
            variance = 0.0

        is_flickering = variance > self.variance_threshold

        if self.smoothed_prob > 0.60 or is_flickering:
            decision = "FAKE"
        else:
            decision = "REAL"

        return {
            "smoothed_probability": self.smoothed_prob,
            "variance": variance,
            "is_flickering": is_flickering,
            "decision": decision
        }

    def reset(self):
        self.history.clear()
        self.smoothed_prob = 0.0