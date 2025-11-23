"""
ML Framework - Core machine learning utilities for trading analysis
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from numba import njit
import warnings

warnings.filterwarnings('ignore')


class FractionalDifferentiation:
    """Fractional differentiation for time series"""

    @staticmethod
    def frac_diff_optimized(series, d, thres=1e-5):
        """
        Fractionally differentiate a time series

        Parameters:
        -----------
        series : array-like
            Time series to differentiate
        d : float
            Differentiation order (0 < d < 1)
        thres : float
            Threshold for weights

        Returns:
        --------
        pd.Series : Fractionally differentiated series
        """
        # Compute weights
        w = [1.0]
        for k in range(1, len(series)):
            w.append(-w[-1] * (d - k + 1) / k)
            if abs(w[-1]) < thres:
                break

        w = np.array(w[::-1])

        # Apply weights
        result = []
        for i in range(len(w) - 1, len(series)):
            result.append(np.dot(w, series[i - len(w) + 1:i + 1]))

        return pd.Series(result, index=range(len(result)))


class StationarityTester:
    """Statistical tests for time series stationarity"""

    @staticmethod
    def adf_test(series, title=''):
        """
        Augmented Dickey-Fuller test for stationarity

        Parameters:
        -----------
        series : array-like
            Time series to test
        title : str
            Title for the test

        Returns:
        --------
        dict : Test results
        """
        result = adfuller(series, autolag='AIC')

        output = {
            'statistic': result[0],
            'pvalue': result[1],
            'usedlag': result[2],
            'nobs': result[3],
            'critical_values': result[4],
            'icbest': result[5]
        }

        return output


class AutoRegressiveModel:
    """Autoregressive model utilities"""

    @staticmethod
    def select_ar_order(series, p_max=50, criterio='aic', limiar=0.01,
                       limiar_pvalor=0.05, min_reducao_absoluta=0.01):
        """
        Select optimal AR order

        Parameters:
        -----------
        series : array-like
            Time series
        p_max : int
            Maximum lag order to test
        criterio : str
            Selection criterion ('aic', 'bic', or 'other')
        limiar : float
            Improvement threshold
        limiar_pvalor : float
            P-value threshold
        min_reducao_absoluta : float
            Minimum absolute reduction

        Returns:
        --------
        int, dict : Optimal order and metrics
        """
        from statsmodels.tsa.ar_model import AutoReg

        best_order = 1
        best_criterion = np.inf
        metrics = {}

        for p in range(1, min(p_max + 1, len(series) // 2)):
            try:
                model = AutoReg(series, lags=p).fit()

                if criterio == 'aic':
                    current_criterion = model.aic
                elif criterio == 'bic':
                    current_criterion = model.bic
                else:
                    current_criterion = np.mean((model.resid) ** 2)

                if current_criterion < best_criterion:
                    improvement = (best_criterion - current_criterion) / best_criterion
                    if improvement > limiar or p == 1:
                        best_criterion = current_criterion
                        best_order = p
                        metrics[p] = {
                            'criterion': current_criterion,
                            'improvement': improvement
                        }
            except:
                continue

        return best_order, metrics


class TripleBarrierMethod:
    """Triple barrier labeling method"""

    @staticmethod
    def applyPtSlOnT1(close, events, ptSl, molecule):
        """
        Apply profit taking and stop loss on time barrier

        Parameters:
        -----------
        close : pd.Series
            Close prices
        events : pd.DataFrame
            Events dataframe with columns: trgt, side, t1
        ptSl : list
            [profit_taking, stop_loss] multipliers
        molecule : list
            Events to process

        Returns:
        --------
        pd.DataFrame : Triple barrier events
        """
        events_ = events.loc[molecule]
        out = events_[['t1']].copy()

        # Initialize columns
        out['sl'] = np.nan
        out['pt'] = np.nan
        out['retorno'] = 0.0
        out['max_drawdown_in_trade'] = 0.0

        for loc, t1 in events_['t1'].items():
            if pd.isnull(t1):
                continue

            df0 = close[loc:t1]
            if len(df0) == 0:
                continue

            # Get target and side
            trgt = events_.at[loc, 'trgt']
            side = events_.at[loc, 'side'] if 'side' in events_.columns else 1

            # Profit taking and stop loss levels
            pt_level = close[loc] + trgt * ptSl[0] * side
            sl_level = close[loc] - trgt * ptSl[1] * side

            # Find first breach
            for dt, price in df0.items():
                ret = (price / close[loc] - 1) * side

                if side == 1:
                    if price >= pt_level:
                        out.at[loc, 'pt'] = dt
                        out.at[loc, 'retorno'] = ret
                        break
                    elif price <= sl_level:
                        out.at[loc, 'sl'] = dt
                        out.at[loc, 'retorno'] = ret
                        break
                else:
                    if price <= pt_level:
                        out.at[loc, 'pt'] = dt
                        out.at[loc, 'retorno'] = ret
                        break
                    elif price >= sl_level:
                        out.at[loc, 'sl'] = dt
                        out.at[loc, 'retorno'] = ret
                        break
            else:
                # Time barrier hit
                out.at[loc, 'retorno'] = (df0.iloc[-1] / close[loc] - 1) * side

        return out

    @staticmethod
    def get_label(row):
        """
        Get label from triple barrier results

        Returns:
        --------
        int : 1 for profit, -1 for loss, 0 for time exit
        """
        if pd.notna(row['pt']):
            return 1
        elif pd.notna(row['sl']):
            return -1
        else:
            return 0


class EventAnalyzer:
    """Event detection and analysis"""

    @staticmethod
    def getVol(close, span0=100):
        """
        Calculate volatility

        Parameters:
        -----------
        close : pd.Series
            Close prices
        span0 : int
            Span for EWM

        Returns:
        --------
        pd.Series : Volatility
        """
        df0 = close.index.to_series().diff()
        df0 = close.pct_change().ewm(span=span0).std()
        return df0

    @staticmethod
    def getTEvents(residuals, vol, h):
        """
        CUSUM filter for event detection

        Parameters:
        -----------
        residuals : pd.Series
            Residuals from model
        vol : pd.Series
            Volatility
        h : float
            Threshold

        Returns:
        --------
        pd.DataFrame : Detected events
        """
        tEvents = []
        sPos = 0
        sNeg = 0

        residuals = residuals.fillna(0)
        vol = vol.fillna(vol.mean())

        for i in range(len(residuals)):
            sPos = max(0, sPos + residuals.iloc[i])
            sNeg = min(0, sNeg + residuals.iloc[i])

            if sPos > h * vol.iloc[i]:
                tEvents.append({
                    'time': residuals.index[i],
                    'trgt': vol.iloc[i],
                    'side': 1
                })
                sPos = 0
            elif sNeg < -h * vol.iloc[i]:
                tEvents.append({
                    'time': residuals.index[i],
                    'trgt': vol.iloc[i],
                    'side': -1
                })
                sNeg = 0

        return pd.DataFrame(tEvents)


class PerformanceAnalyzer:
    """Performance metrics and analysis"""

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate performance metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        return {'mae': mae, 'rmse': rmse, 'r2': r2}


class ResidualAnalyzer:
    """Residual analysis utilities"""

    @staticmethod
    def analyze_residuals(residuals):
        """Analyze residuals"""
        return {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skew': pd.Series(residuals).skew(),
            'kurtosis': pd.Series(residuals).kurtosis()
        }


class DataDownloader:
    """Data downloading utilities (placeholder)"""
    pass


class DollarBarsProcessor:
    """Dollar bars processing (placeholder)"""
    pass


class TripleBarrierExecutor:
    """Triple barrier execution (placeholder)"""
    pass


def zscore_normalize(series):
    """Z-score normalization"""
    return (series - series.mean()) / series.std()


def av_error(y_true, y_pred):
    """Average error"""
    return np.mean(np.abs(y_true - y_pred))
