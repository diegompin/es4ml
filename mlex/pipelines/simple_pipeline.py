import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import (
    Pipeline, 
    FeatureUnion
) 
from functools import reduce
from mlex import (
    NumericalTransfomer,
    CategoricalOneHotTransfomer,
    CompositeTranformer
)

from mlex import (
    SequenceTransfomer
)

from mlex import(
    PastFutureSplit
)

class SimplePipeline(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 numeric_features, 
                 categorical_features,
                 final_model,   #o que seria esse final mode? um exemplo?
                 epochs=10, #input_shape
                 ) -> None:
        super().__init__()
        self.numberic_feature = numeric_features
        self.categorical_features = categorical_features
        self.final_model = final_model
        self.model = self._build_model()

    @property
    def name(self):
        return __name__
        
    def fit(self, X, y=None):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score_samples(self, X):
        return self.model.score_samples(X=X)

    def _build_model(self):
        
        data_extraction = PastFutureSplit.csv_extraction(
            "/data/pcpe_01.csv"
        )

        preprocessor = CompositeTranformer(
            numeric_features=self.numberic_feature, 
            categorical_features=self.categorical_features
        )
        
        self.final_model.build()
        self.final_model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=[
                        'acc', 
                        #TODO evaluate the need of this AUC here
                        # tf.keras.metrics.AUC()
                        ])
        sequence = SequenceTransfomer()
        model = Pipeline(
            steps=[
                ("Data Extraction", data_extraction),
                ("preprocessing", preprocessor),
                ("sequence", sequence),
                ("final_model", self.final_model),
                ]
            )
        
        return model
