import keras
import keras.layers
import keras.optimizers.adam
import tensorflow as tf

import abc

class BaseModel(abc.ABC):

    @abc.abstractmethod
    def build_model(self):
        pass
    
    def get_model(self) -> keras.Sequential:
        self.build_model()
        self.model.build()  
        self.compile()
        return self.model
        

    def compile(self):
        self.model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc', tf.keras.metrics.AUC()])
    
    def summary(self):
        return self.model.summary()
    
    def fit(self, X, y=None):
        return self.model.fit(X, epochs=10)
    
    def predict(self,X, y=None):
        return self.model.predict(X)

class SimpleRNNModel(BaseModel):

    def __init__(self, input_shape) -> None:
        super().__init__()
        self.input_shape = input_shape

    def build_model(self) -> keras.Sequential:
        self.model = keras.models.Sequential([
            keras.layers.SimpleRNN(16,  return_sequences=True, input_shape=self.input_shape),
            keras.layers.SimpleRNN(16),
            keras.layers.Dense(1, activation='sigmoid')
        ])    
   
        
class SimpleLSTMModel(BaseModel):

    def __init__(self,input_shape)-> None:
        super().__init__()
        self.input_shape = input_shape

    def build_model(self) -> keras.Sequential:

        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(16, return_sequences=True, input_shape=self.input_shape),
             tf.keras.layers.LSTM(16),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
   
    
class SimpleGruModel(BaseModel):
    
    def __init__(self,input_shape)-> None:
        super().__init__()
        self.input_shape = input_shape

    def build_model(self) ->keras.Sequential:
        self.model = tf.keras.Sequential([
            tf.keras.layers.GRU(16, return_sequences=True, input_shape = self.input_shape),
            tf.keras.layers.GRU(16),
            tf.keras.layers.Dense(1,activation='sigmoid')
        ])