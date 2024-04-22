
from mlex import SimplePipeline
from mlex import SimpleRNNModel

from mlex import PastFutureSplit
splitter = PastFutureSplit()

proportion = 0.5
X_train, y_train, X_test, y_test = splitter.train_test_split(proportion=proportion)
dimensions = X_train.shape[-1]

models = [
    SimpleRNNModel(input_shape=[None, dimensions]),
    # SimpleGRUModel(input_shape=[None, dimensions]),
]

for model in models:
    pipeline = SimplePipeline(final_model=model)
    history = pipeline.fit(X_train, y_train, epochs=10)
    y_pred = pipeline.predict(X_test, y_test)

    