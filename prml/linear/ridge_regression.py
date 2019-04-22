import numpy as np


class RidgeRegression:
    """
    Ridge regression model
    w* = argmin |t - X @ w|^2 + alpha * |w|_2^2
    """

    def __init__(self, alpha:float=1.):
        self.alpha = alpha

    def fit(self, X:np.ndarray, t:np.ndarray):
        """
        maximum a posteriori estimation of parameter
        Parameters
        ----------
        X : (N, D) np.ndarray
            training data independent variable
        t : (N,) np.ndarray
            training data dependent variable
        """

        eye = np.eye(X.shape[1])
        self.w = np.linalg.solve(self.alpha * eye + X.T @ X, X.T @ t)
        self.coef_ = self.w

    def predict(self, X:np.ndarray):
        """
        make prediction given input
        Parameters
        ----------
        X : (N, D) np.ndarray
            samples to predict their output
        Returns
        -------
        (N,) np.ndarray
            prediction of each input
        """
        return X @ self.w