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


from sklearn.metrics import confusion_matrix
from mlex.utils.splits import PastFutureSplit

from mlex.models.models import SimpleRNNModel

from sklearn import metrics

class SimplePipeline(BaseEstimator, ClassifierMixin):

    def __init__(self,
                #  numeric_features, 
                #  categorical_features,
                #  X,
                #  y,
                 final_model,
                 epochs=10, 
                ) -> None:
        super().__init__()
        # self.numeric_features = numeric_features
        # self.categorical_features = categorical_features
        # self.X = X
        # self.y = y
        self.final_model = final_model
        self.epochs = epochs
        self.pipeline = self._build_pipeline()

    @property
    def name(self):
        return __name__
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y, 
                                 final_model__epochs = self.epochs)

    def predict(self, X):
        return self.pipeline.predict(X)

    def score_samples(self, X):
        return self.pipeline.score_samples(X=X)
    
   
    # def make_sequence(self)->None:
    #     sequence = SequenceTransfomer()
    #     self.X_train = np.array(self.X_train)
    #     self.y_train = np.array(self.y_train)
    #     self.data_train = sequence.transform(
    #         X = self.X_train,
    #         y = self.y_train
    #     )

    #     self.X_test = np.array(self.X_test)
    #     self.y_test = np.array(self.y_test)
        
    #     self.data_test = sequence.transform(
    #         X = self.X_test,
    #         y = self.y_test
    #     )

    def _build_pipeline(self):
        
        # self.final_model.get_model()
        #  

        # preprocessor = CompositeTranformer(
        #     numeric_features=self.numeric_features, 
        #     categorical_features=self.categorical_features
        # )

        # Xt = preprocessor.transform(self.X)

       

         
        """self.final_model.build()
        self.final_model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=[
                        'acc', 
                       TODO evaluate the need of this AUC here
                        tf.keras.metrics.AUC()
                        ])
       """

        
        # sequence  = self.make_sequence()
        sequence = SequenceTransfomer()
        
        # self.history = self.final_model.fit(self.data_train)

        # y_pred = self.final_model.predict(self.data_test)


        # conf_matrix = self.plot_matrix(y_test = self.y_test, y_pred = y_pred)
        
        # roc = self.plot_graphic(y_test = self.y_test, y_pred = y_pred)

        pipeline = Pipeline(
            steps=[
                # ("preprocessing", preprocessor),
                # ("Data Split", split),
                # ("sequence", sequence),
                ("final_model", self.final_model),
                # ("Fitting", self.final_model),
                # ("Predicting", self.final_model),
                # ("matrix confusion", conf_matrix),
                # ("ROC", roc)
                ]
            )
        
        return pipeline
