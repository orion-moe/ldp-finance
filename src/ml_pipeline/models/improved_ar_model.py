"""
Improved AR Model - Autoregressive model with multicollinearity treatment
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class ImprovedAutoRegressiveModel:
    """
    Improved Autoregressive Model with automatic multicollinearity treatment

    This class implements AR models with various regularization techniques
    to handle multicollinearity issues.
    """

    def __init__(self, series):
        """
        Initialize the model

        Parameters:
        -----------
        series : array-like
            Time series data
        """
        self.series = np.array(series)
        self.model = None
        self.results = None

    def fit_with_multicollinearity_treatment(self, p, treatment_method='auto', alpha=1.0):
        """
        Fit AR model with multicollinearity treatment

        Parameters:
        -----------
        p : int
            AR order
        treatment_method : str
            Method to use: 'ols', 'ridge', 'lasso', 'elastic_net', 'auto'
        alpha : float
            Regularization strength

        Returns:
        --------
        dict : Model results
        """
        # Create lagged features
        X, y = self._create_lagged_features(p)

        if treatment_method == 'auto':
            # Try OLS first, fall back to Ridge if issues
            try:
                return self._fit_ols(p, X, y)
            except:
                print("   OLS failed, using Ridge regression")
                return self._fit_ridge(p, X, y, alpha)

        elif treatment_method == 'ols':
            return self._fit_ols(p, X, y)

        elif treatment_method == 'ridge':
            return self._fit_ridge(p, X, y, alpha)

        elif treatment_method == 'lasso':
            return self._fit_lasso(p, X, y, alpha)

        elif treatment_method == 'elastic_net':
            return self._fit_elastic_net(p, X, y, alpha)

        else:
            raise ValueError(f"Unknown treatment method: {treatment_method}")

    def _create_lagged_features(self, p):
        """Create lagged features for AR model"""
        n = len(self.series)
        X = np.zeros((n - p, p))
        y = self.series[p:]

        for i in range(p):
            X[:, i] = self.series[p - i - 1:n - i - 1]

        return X, y

    def _fit_ols(self, p, X, y):
        """Fit using OLS (via statsmodels AutoReg)"""
        model = AutoReg(self.series, lags=p)
        results = model.fit()

        # Extract parameters
        params = results.params[1:]  # Exclude constant
        constant = results.params[0]

        # Predictions
        y_pred = results.predict(start=p, end=len(self.series) - 1)

        # Residuals
        residuals = self.series[p:] - y_pred

        # Metrics
        metrics = {
            'mae': mean_absolute_error(self.series[p:], y_pred),
            'rmse': np.sqrt(mean_squared_error(self.series[p:], y_pred)),
            'r2': r2_score(self.series[p:], y_pred)
        }

        return {
            'method': 'ols',
            'p': p,
            'params': params,
            'constant': constant,
            'y_pred': y_pred,
            'residuals': residuals,
            'metrics': metrics,
            'model': results
        }

    def _fit_ridge(self, p, X, y, alpha):
        """Fit using Ridge regression"""
        model = Ridge(alpha=alpha)
        model.fit(X, y)

        # Predictions
        y_pred = model.predict(X)

        # Residuals
        residuals = y - y_pred

        # Metrics
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred)
        }

        return {
            'method': 'ridge',
            'p': p,
            'params': model.coef_,
            'constant': model.intercept_,
            'y_pred': y_pred,
            'residuals': residuals,
            'metrics': metrics,
            'model': model
        }

    def _fit_lasso(self, p, X, y, alpha):
        """Fit using Lasso regression"""
        model = Lasso(alpha=alpha)
        model.fit(X, y)

        y_pred = model.predict(X)
        residuals = y - y_pred

        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred)
        }

        return {
            'method': 'lasso',
            'p': p,
            'params': model.coef_,
            'constant': model.intercept_,
            'y_pred': y_pred,
            'residuals': residuals,
            'metrics': metrics,
            'model': model
        }

    def _fit_elastic_net(self, p, X, y, alpha):
        """Fit using Elastic Net regression"""
        model = ElasticNet(alpha=alpha)
        model.fit(X, y)

        y_pred = model.predict(X)
        residuals = y - y_pred

        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred)
        }

        return {
            'method': 'elastic_net',
            'p': p,
            'params': model.coef_,
            'constant': model.intercept_,
            'y_pred': y_pred,
            'residuals': residuals,
            'metrics': metrics,
            'model': model
        }

    def monteCarlo_improved(self, series_oos, ar_results):
        """
        Make predictions on out-of-sample data

        Parameters:
        -----------
        series_oos : array-like
            Out-of-sample series
        ar_results : dict
            Results from fit_with_multicollinearity_treatment

        Returns:
        --------
        pd.DataFrame : Predictions
        """
        series_oos = np.array(series_oos)
        p = ar_results['p']
        params = ar_results['params']
        constant = ar_results['constant']

        predictions = []

        for i in range(len(series_oos)):
            if i < p:
                # Use actual values for initial lags
                pred = constant
                for j in range(i + 1):
                    if j < len(params):
                        pred += params[j] * series_oos[i - j]
            else:
                # Use lagged values
                pred = constant
                for j in range(p):
                    pred += params[j] * series_oos[i - j - 1]

            predictions.append(pred)

        return pd.DataFrame({
            'fraqdiff_pred': predictions
        })
