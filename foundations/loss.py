import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        # - y_true * log(y_pred) + (1-y_true) * log(1-y_pred)

        y_pred += np.finfo(float).eps
        # print(y_true, 1-y_true)

        # print(- (y_true * np.log(y_pred) + ((1-y_true) * np.log(1-y_pred))))
        ans = - ( y_true * np.log(y_pred) + ((1-y_true) * np.log(1-y_pred)))
        # print(sum(ans), len(y_pred))
        ans = np.mean(ans)
        # print(sum(ans)/len(y_pred))
        return np.round(ans, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        y_pred += np.finfo(float).eps
        ans = -np.mean(np.sum(y_true * np.log(y_pred), axis = 1))
        # print(np.log(y_pred))
        # print(y_true * y_pred)
        # print(-np.log(y_true * y_pred))
        print(ans)
        # print(y_pred)
        return round(ans, 4)
