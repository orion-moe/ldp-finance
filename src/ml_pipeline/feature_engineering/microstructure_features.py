"""
Microstructure Features - Market microstructure indicators
"""

import numpy as np
import pandas as pd
from numba import njit


class MicrostructureFeatures:
    """Basic microstructure feature calculations"""

    @staticmethod
    def calculate_roll_spread(prices, window=100):
        """
        Calculate Roll's spread estimator

        Parameters:
        -----------
        prices : pd.Series
            Price series
        window : int
            Window size

        Returns:
        --------
        pd.Series : Roll spread
        """
        returns = prices.pct_change()

        def roll_estimator(rets):
            if len(rets) < 2:
                return np.nan
            cov = np.cov(rets[:-1], rets[1:])[0, 1]
            if cov >= 0:
                return 0
            return 2 * np.sqrt(-cov)

        roll_spread = returns.rolling(window).apply(roll_estimator, raw=True)
        return roll_spread

    @staticmethod
    def calculate_amihud_illiquidity(prices, volumes, window=100):
        """
        Calculate Amihud illiquidity measure

        Parameters:
        -----------
        prices : pd.Series
            Price series
        volumes : pd.Series
            Volume series
        window : int
            Window size

        Returns:
        --------
        pd.Series : Amihud illiquidity
        """
        returns = prices.pct_change().abs()
        illiq = returns / (volumes + 1e-10)
        return illiq.rolling(window).mean()


class CryptoMicrostructureAnalysis:
    """
    Crypto-specific microstructure analysis

    This class handles the calculation of microstructure features
    for cryptocurrency data.
    """

    def __init__(self, symbols, windows=[50, 100]):
        """
        Initialize the analyzer

        Parameters:
        -----------
        symbols : list
            List of symbol names
        windows : list
            List of window sizes for calculations
        """
        self.symbols = symbols
        self.windows = windows
        self.data = {}
        self.features = {}

    def load_data(self, data_dict):
        """
        Load data for analysis

        Parameters:
        -----------
        data_dict : dict
            Dictionary with symbol as key and DataFrame as value
        """
        self.data = data_dict

    def calculate_all_features(self):
        """
        Calculate all microstructure features
        """
        for symbol in self.symbols:
            if symbol not in self.data:
                continue

            df = self.data[symbol]

            # Rename columns if needed
            if 'fraq_close' in df.columns:
                price_col = 'fraq_close'
            elif 'close' in df.columns:
                price_col = 'close'
            else:
                continue

            prices = df[price_col]
            volumes = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

            # Initialize features dataframe
            features_df = pd.DataFrame(index=df.index)

            # Calculate features for each window
            for window in self.windows:
                # Roll spread
                features_df[f'roll_spread_{window}'] = MicrostructureFeatures.calculate_roll_spread(
                    prices, window
                )

                # Amihud illiquidity
                features_df[f'amihud_{window}'] = MicrostructureFeatures.calculate_amihud_illiquidity(
                    prices, volumes, window
                )

                # Volatility
                features_df[f'volatility_{window}'] = prices.pct_change().rolling(window).std()

            self.features[symbol] = features_df

    def get_features(self, symbol):
        """
        Get features for a symbol

        Parameters:
        -----------
        symbol : str
            Symbol name

        Returns:
        --------
        pd.DataFrame : Features
        """
        return self.features.get(symbol, pd.DataFrame())
