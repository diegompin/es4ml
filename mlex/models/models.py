import keras
import keras.layers
import keras.optimizers.adam
import tensorflow as tf

import abc

class BaseModel(abc.ABC):

    @abc.abstractmethod
    def get_model():
        pass

    @abc.abstractmethod
    def compile():
        pass

    @abc.abstractmethod
    def summary():
        pass

    @abc.abstractmethod
    def fit(X, y=None):
        pass


class SimpleRNNModel(BaseModel):

    def __init__(self, input_shape) -> None:
        super().__init__()
        self.input_shape = input_shape

    def get_model(self) -> keras.Sequential:
        self.model = keras.models.Sequential([
            # keras.layers.SimpleRNN(),
            # keras.layers.GRU(32, dropout=0.1, recurrent_dropout=.5,  input_shape=[None, X.shape[-1]]),
            # keras.layers.Conv1D(8, 6, input_shape=[None, X.shape[-1]]),
            keras.layers.SimpleRNN(16,  return_sequences=True, input_shape=self.input_shape),
            # keras.layers.LSTM(16,  return_sequences=True),
            keras.layers.SimpleRNN(16, ),
            # keras.layers.LSTM(32, dropout=0.1, recurrent_dropout=.1),
            # keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.build()
        return self.model
    

    def compile(self):
        return self.model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc', tf.keras.metrics.AUC()])
    
    def summary(self):
        return self.model.summary()
    
    def fit(self, X, y=None):
        return self.model.fit(X, epochs=10)
    
    
class SimpleLSTMModel(BaseModel):

    def __init__(self,input_shape)-> None:
        super().__init__()
        self.input_shape = input_shape

    def get_model(self) -> keras.Sequential:
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(16, return_sequences=True, input_shape=self.input_shape),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.build()
        return model
    
    def compile(self):
        return self.model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc', tf.keras.metrics.AUC()])
    
    def summary(self):
        return self.model.summary()
    
    def fit(self, X, y=None):
        return self.model.fit(X, epochs=10)
    
    
class SimpleGruModel(BaseModel):

    def __init__(self,input_shape)-> None:
        super().__init__()
        self.input_shape = input_shape

    def get_mode(self) -> keras.Sequential:
        model = tf.keras.Sequential([
            tf.keras.layers.GRU(16, return_sequences=True, input_shape = self.input_shape),
            tf.keras.layers.Dense(1,activation='sigmoid')
        ])
        self.model.build()
        return model
    
    def compile(self):
        return self.model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc', tf.keras.metrics.AUC()])
    
    def summary(self):
        return self.model.summary()
    
    def fit(self, X, y=None):
        return self.model.fit(X, epochs=10)