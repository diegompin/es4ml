import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from functools import reduce
import itertools as ite

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class CategoricalOneHotTransfomer(BaseEstimator, TransformerMixin):
    
    def __init__(self) -> None:
        super().__init__()
        self.encoder = OneHotEncoder()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = self.encoder.fit_transform(X)
        return Xt
    

class NumericalTransfomer(BaseEstimator, TransformerMixin):
    
    def __init__(self) -> None:
        super().__init__()
        self.encoder = MinMaxScaler()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = self.encoder.fit_transform(X)
        return Xt
    
class CompositeTranformer(BaseEstimator, TransformerMixin):
     
    def __init__(self, numeric_features, categorical_features) -> None:
        super().__init__()
        self.numberic_feature = numeric_features
        self.categorical_features = categorical_features
        self.encoder =  ColumnTransformer(    #o que seria Column transformer?
            transformers=[
                ("num", NumericalTransfomer(), self.numberic_feature),
                ("cat", CategoricalOneHotTransfomer(), self.categorical_features),
            ]
        )
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = self.encoder.fit_transform(X, y)
        return Xt

#TODO implementar
class EmbeedinglTransfomer(BaseEstimator, TransformerMixin):
    
    def __init__(self) -> None:
        super().__init__()
        # self.encoder = Embeding(handle_unknown="ignore")
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        # Xt = self.encoder.fit_transform(X)
        return X