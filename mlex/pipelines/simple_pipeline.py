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

from mlex.features.columns import (
    CompositeTranformer,
)

from mlex.features.sequences import (
    SequenceTransfomer
)


class SimplePipeline(BaseEstimator, ClassifierMixin):

    def __init__(self, numeric_features, categorical_features, final_model, epochs = 10) -> None:
        super().__init__()
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.final_model = final_model
        self.epochs = epochs

    @property
    def name(self):
        return __name__
        
    def fit(self, X, ep):
        return self.final_model.fit(X, y)

    def predict(self, X):
        return self.final_model.predict(X)

    def score_samples(self, X):
        return self.final_model.score_samples(X=X)

    def _build_model(self):
        pass
    

        preprocessor = CompositeTranformer(
            numeric_features=self.s, 
            categorical_features=self.categorical_features
        )
            

        sequence = SequenceTransfomer(
                                    numeric_features=self.numeric_features, 
                                    categorical_features=self.categorical_features
                                    )


        final_model_build = self.final_model.build()
        final_model_compile = self.final_model.compile(
                                loss='binary_crossentropy',
                                optimizer='rmsprop',
                                metrics=['acc', tf.keras.metrics.AUC()]
                                )
        
        final_model_summary = self.final_model.summary()
       
        model = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("sequence", sequence),
                ("final_model", self.final_model),
                ("build", final_model_build),
                ("compile", final_model_compile),
                ("summary", final_model_summary)
                ]
            )
        
        
        return model