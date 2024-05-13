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
    
    def plot_matrix(self,y_test,y_pred)->None: 
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

    def plot_graphic(self,y_test,y_pred)->None:
        sequence_length = 5

        title = "ROC"
        fpr, tpr, thresholds = metrics.roc_curve(y_test[sequence_length:-sequence_length+1], y_pred)
        auc = metrics.auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='deeppink', linewidth=4, label=f"ROC Curve (area = {round(auc,2) })")
        ax.plot([0,1], [0,1], "k--",linewidth=4, label='random classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=16)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=16)
        ax.set_title(f"Receiver Operating Characteristic \n {title}", fontsize=18)
        ax.legend(loc="lower right")
        plt.savefig("../results/roc.pdf")
        plt.show()

    def make_sequence(self)->None:
        sequence = SequenceTransfomer()
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.data_train = sequence.transform(
            X = self.X_train,
            y = self.y_train
        )

        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
        
        self.data_test = sequence.transform(
            X = self.X_test,
            y = self.y_test
        )

    def _build_model(self):
    

        preprocessor = CompositeTranformer(
            numeric_features=self.numberic_feature, 
            categorical_features=self.categorical_features
        )

        Xt = preprocessor.transform(self.df)

        split = PastFutureSplit()
        self.X_train, self.X_test, self.y_train, self.y_test = split.train_test_split(Xt,self.y)

         
        """self.final_model.build()
        self.final_model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=[
                        'acc', 
                       TODO evaluate the need of this AUC here
                        tf.keras.metrics.AUC()
                        ])
       """

        
        sequence  = self.make_sequence()
        

        #isso aqui, pode tirar do pipeline
        self.final_model = SimpleRNNModel(input_shape=Xt.shape)
        self.final_model.get_model()
        self.final_model.compile()
        
        self.history = self.final_model.fit(self.data_train)

        y_pred = self.final_model.predict(self.data_test)


        conf_matrix = self.plot_matrix(y_test = self.y_test, y_pred = y_pred)
        
        roc = self.plot_graphic(y_test = self.y_test, y_pred = y_pred)

        model = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("Data Split", split),
                ("sequence", sequence),
                ("final_model", self.final_model),
                ("Fitting", self.final_model),
                ("Predicting", self.final_model),
                ("matrix confusion", conf_matrix),
                ("ROC", roc)
                ]
            )
        
        return model
