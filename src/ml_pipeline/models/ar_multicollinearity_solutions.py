"""
AR Multicollinearity Solutions - Methods to handle multicollinearity in AR models
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from statsmodels.stats.outliers_influence import variance_inflation_factor


class ARMulticollinearityTreatment:
    """
    Methods to diagnose and treat multicollinearity in AR models

    Provides various techniques including:
    - VIF (Variance Inflation Factor) analysis
    - Ridge regression
    - Lasso regression
    - Elastic Net
    - PCA
    - Stepwise selection
    """

    @staticmethod
    def calculate_vif(X):
        """
        Calculate Variance Inflation Factor for features

        Parameters:
        -----------
        X : np.array or pd.DataFrame
            Feature matrix

        Returns:
        --------
        pd.DataFrame : VIF values
        """
        vif_data = pd.DataFrame()
        vif_data["feature"] = range(X.shape[1])
        vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

        return vif_data

    @staticmethod
    def detect_multicollinearity(X, threshold=10):
        """
        Detect multicollinearity using VIF

        Parameters:
        -----------
        X : np.array
            Feature matrix
        threshold : float
            VIF threshold (typically 10)

        Returns:
        --------
        dict : Diagnosis results
        """
        vif = ARMulticollinearityTreatment.calculate_vif(X)

        has_multicollinearity = (vif["VIF"] > threshold).any()

        return {
            'has_multicollinearity': has_multicollinearity,
            'vif': vif,
            'max_vif': vif["VIF"].max(),
            'problematic_features': vif[vif["VIF"] > threshold]["feature"].tolist()
        }

    @staticmethod
    def apply_pca(X, n_components=None, variance_threshold=0.95):
        """
        Apply PCA to reduce multicollinearity

        Parameters:
        -----------
        X : np.array
            Feature matrix
        n_components : int
            Number of components (if None, use variance_threshold)
        variance_threshold : float
            Cumulative variance threshold

        Returns:
        --------
        dict : PCA results
        """
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(X)

        if n_components is None:
            # Find number of components for variance threshold
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components_needed = np.argmax(cumsum_variance >= variance_threshold) + 1

            pca = PCA(n_components=n_components_needed)
            X_transformed = pca.fit_transform(X)

        return {
            'X_transformed': X_transformed,
            'pca': pca,
            'n_components': pca.n_components_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
        }

    @staticmethod
    def stepwise_selection(X, y, significance_level=0.05):
        """
        Stepwise feature selection to remove collinear features

        Parameters:
        -----------
        X : np.array
            Feature matrix
        y : np.array
            Target variable
        significance_level : float
            P-value threshold

        Returns:
        --------
        list : Selected feature indices
        """
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant

        n_features = X.shape[1]
        selected_features = []

        for i in range(n_features):
            best_pval = 1.0
            best_feature = None

            # Try adding each remaining feature
            for j in range(n_features):
                if j in selected_features:
                    continue

                test_features = selected_features + [j]
                X_test = add_constant(X[:, test_features])

                try:
                    model = OLS(y, X_test).fit()
                    pval = model.pvalues[-1]  # P-value of the new feature

                    if pval < best_pval and pval < significance_level:
                        best_pval = pval
                        best_feature = j
                except:
                    continue

            if best_feature is not None:
                selected_features.append(best_feature)
            else:
                break

        return selected_features

    @staticmethod
    def recommend_treatment(X, y):
        """
        Recommend treatment based on multicollinearity diagnosis

        Parameters:
        -----------
        X : np.array
            Feature matrix
        y : np.array
            Target variable

        Returns:
        --------
        str : Recommended treatment method
        """
        diagnosis = ARMulticollinearityTreatment.detect_multicollinearity(X)

        if not diagnosis['has_multicollinearity']:
            return 'ols'  # No treatment needed

        max_vif = diagnosis['max_vif']

        if max_vif < 20:
            return 'ridge'  # Mild multicollinearity
        elif max_vif < 100:
            return 'elastic_net'  # Moderate multicollinearity
        else:
            return 'pca'  # Severe multicollinearity
