
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from functools import reduce
import itertools as ite
import tensorflow as tf

class SequenceTransfomer(BaseEstimator, TransformerMixin):
    
    def __init__(self, 
                 sequence_length = 10,
                 batch_size = 128
                 ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.batch_size = batch_size
       
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y, **fit_params):
        data_train = tf.keras.utils.timeseries_dataset_from_array(
            # X[:-self.sequence_length],
            # y[self.sequence_length:],
            X,
            y,
            sequence_length=self.sequence_length,
            sequence_stride=1,
            sampling_rate=1,
            batch_size=self.batch_size,
            shuffle=False,
            seed=None,
            start_index=None,
            end_index=None,
        )
        return data_train




