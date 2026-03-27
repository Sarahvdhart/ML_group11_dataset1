#Importing functions
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

#Make a class to be used in the pipeline
class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, zero_threshold=0.90, clip_iqr=True, corr_threshold=0.85): 
        self.zero_threshold = zero_threshold #Drop feature if 0 ratio is above threshold
        self.clip_iqr = clip_iqr #Perform outlier clipping based on IQR
        self.corr_threshold = corr_threshold #Drop features with correlation above threshold
        self.scaler = RobustScaler()

    #Function Fit to determine which features to drop
    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        #Drop features with too many zeros
        zero_ratio = (X == 0).mean()
        keep_zero = zero_ratio <= self.zero_threshold
        
        #Drop features with zero variance 
        variance = X.var()
        keep_var = variance > 0
        
        self.keep_columns_ = X.columns[keep_zero & keep_var]
        X = X[self.keep_columns_]

        #Correlation filter
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        self.high_corr_features_ = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
        
        self.selected_columns_ = [col for col in X.columns if col not in self.high_corr_features_]

        X = X[self.selected_columns_]

        #Outlier removal
        if self.clip_iqr:
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            self.lower_ = Q1 - 1.5 * IQR
            self.upper_ = Q3 + 1.5 * IQR

        self.scaler.fit(X)
        return self

    #Define function transform that applies transformation to data
    def transform(self, X):
        X = pd.DataFrame(X)
        X = X[self.keep_columns_] #Keep columns selected during fitting
        X = X[self.selected_columns_] #Remove highly correlated features

        if self.clip_iqr: #Outlier clipping
            X = X.clip(self.lower_, self.upper_, axis=1)

        X_scaled = self.scaler.transform(X) 

        return X_scaled
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_columns_)
    

