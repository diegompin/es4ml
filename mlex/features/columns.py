import abc
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from functools import reduce
import itertools as ite

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class CustomTrasformer(BaseEstimator, TransformerMixin, abc.ABC):

    @abc.abstractmethod
    def fit(X, y=None):
        pass

    @abc.abstractmethod
    def transform(X, y=None, **fit_params):
        pass

class CategoricalOneHotTransfomer(CustomTrasformer):
    
    def __init__(self) -> None:
        super().__init__()
        self.encoder = OneHotEncoder()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = self.encoder.fit_transform(X)
        return Xt
    

class NumericalTransfomer(CustomTrasformer):
    
    def __init__(self) -> None:
        super().__init__()
        self.scaler = MinMaxScaler()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = self.scaler.fit_transform(X)
        return Xt
    
class CompositeTranformer(CustomTrasformer):
     
    def __init__(self, numeric_features, categorical_features) -> None:
        super().__init__()
        self.numeric_feature = numeric_features
        self.categorical_features = categorical_features
        self.encoder =  ColumnTransformer( 
            transformers=[
                ("num", NumericalTransfomer(), self.numeric_feature),
                ("cat", CategoricalOneHotTransfomer(), self.categorical_features),
            ]
        )
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        Xt = np.column_stack([ NumericalTransfomer().fit_transform, CategoricalOneHotTransfomer().fit_transform])
        return Xt


class EmbeedinglTransfomer(CustomTrasformer):
    
    def __init__(self) -> None:
        super().__init__()
        # self.encoder = Embedding(handle_unknown="ignore")
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        # Xt = self.encoder.fit_transform(X)
        return X