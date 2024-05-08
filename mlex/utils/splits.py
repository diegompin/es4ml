import pandas as pd

class PastFutureSplit:

    def __init__(self) -> None:
        super().__init__()

    def csv_extraction(self,path):
        df = pd.read_csv(path, delimiter=';')
        return df

    def train_test_split(self, X, y, proportion=.5):
        mid = int(proportion *len(X))
        X_train = X[:mid]
        y_train = y[:mid]
        X_test = X[mid:-1]
        y_test = y[mid:-1]
        return X_train, X_test, y_train, y_test 