import pandas as pd
import numpy as np
from random import random
from typing import Optional, Union
import math


class LogisticRegressor():

    def __init__(self, x_train: list, y_train: list, step_size: int, num_iter: int):
        """
        Pass in parameters during initialization
        """
        self.x_train = x_train
        self.y_train = y_train
        self.step_size = step_size
        self.num_iter = num_iter

        if x_train is None or y_train is None or len(x_train) == 0 or len(x_train) != len(y_train):
            raise Exception("Training X and Labels have unequal length")

        self.num_features = len(x_train[0])

    def train(self) -> None:
        """
        Train the model. Loss function used is Log Loss.
        Gradient of Loss function w.r.t features is x_train . (sigmoid(Theta.x) - y)
        """
        self.theta = [random() for _ in range(self.num_features)]

        if not self._at_stopping_point():
            self.num_iter -= 1

            # Get predictions and loss
            y_pred = [self._sigmoid(pred) for pred in np.dot(self.x_train, self.theta)]

            loss = np.subtract(y_pred, self.y_train)

            # Calculate gradient
            gradient = np.dot(np.array(self.x_train).T, loss)

            # Update weights
            self.theta = np.subtract(self.theta, np.multiply(self.step_size, gradient))

    def predict(self, x: list) -> Optional[list]:
        if not self.theta:
            raise Exception("Need to train model before running predictions.")

        """
        Take a dot product of the features with theta and pass the result through sigmoid.
        """
        y_hat = []
        for x_pred in x:
            y_pred = np.dot(self.theta, x_pred)
            y_hat.append(self._sigmoid(y_pred))

        return y_hat

    def _at_stopping_point(self) -> bool:
        """
        stopping point is at num_iter iterations.
        """
        return self.num_iter == 0

    def _sigmoid(self, val: Union[int, float]) -> float:
        return math.exp(val) / (1 + math.exp(val))
