import numpy as np
import pandas as pd


class Perceptron:
    def __init__(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.x = x
        self.y = y
        self.preprocess()

    def preprocess(self):
        if len(self.y.unique() == 2):
            self.y[self.y == self.y.unique()[0]] = +1
            self.y[self.y == self.y.unique()[1]] = -1
        else:
            raise NotImplementedError

        cols = self.x.columns
        self.x["intercept"] = 1.0
        new_cols = ["intercept"]
        new_cols.extend(cols)
        self.x = self.x[new_cols]

        self.y = self.y.to_numpy()
        self.x = self.x.to_numpy()
        self.w = np.random.rand(self.x.shape[1])

    def fit(self, iteration=1000, lr=1):
        for _ in range(iteration):
            for index in range(len(self.x)):
                if self.y[index] == +1 and np.dot(self.w.T, self.x[index]) < 0:
                    self.w = self.w + lr * self.x[index]

                if self.y[index] == -1 and np.dot(self.w.T, self.x[index]) > 0:
                    self.w = self.w - lr * self.x[index]
