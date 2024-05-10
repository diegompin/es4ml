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

from mlex.pipelines.simple_pipeline import SimplePipeline
from mlex.models.models import SimpleRNNModel


class SimplePipeline(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 numeric_features, 
                 categorical_features,
                 df,
                 y,
                 epochs=10, 
                ) -> None:
        super().__init__()
        self.numberic_feature = numeric_features
        self.categorical_features = categorical_features
        self.df = df
        self.y = y
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
    

        preprocessor = CompositeTranformer(
            numeric_features=self.numberic_feature, 
            categorical_features=self.categorical_features
        )

        Xt = preprocessor.transform(self.df)

        split = PastFutureSplit()
        X_train, X_test, y_train, y_test = split.train_test_split(Xt,self.y)


        
        """self.final_model.build()
        self.final_model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=[
                        'acc', 
                       TODO evaluate the need of this AUC here
                        tf.keras.metrics.AUC()
                        ])
       """
        
        split = PastFutureSplit()
        X_train, X_test, y_train, y_test = split.train_test_split(Xt,y)

        sequence = SequenceTransfomer()

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        data_train = sequence.transform(
            X = X_train,
            y = y_train
        )

        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        data_test = sequence.transform(
            X = X_train,
            y = y_train
        )


        self.final_model = SimpleRNNModel(input_shape=Xt.shape)
        self.final_model.get_model()
        self.final_model.compile()
        
        history = self.final_model.fit(data_train)

        y_pred = self.final_model.predict(data_test)


        sequence_length = 5


        conf_matrix = confusion_matrix(y_true=y_test[sequence_length:-sequence_length+1], y_pred=y_pred > np.quantile(y_pred, 0.95))

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.savefig('../results/confusion_matrix.pdf')
        plt.show()
        model = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("Data Split", split),
                ("sequence", sequence),
                ("final_model", self.final_model),
                ("Fitting", self.final_model),
                ("Predicting", self.final_model),
                ("matrix confusion", conf_matrix),
                ]
            )
        
        return model
