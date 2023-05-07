import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

class Preprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.imputer = KNNImputer(n_neighbors = 2)
        self.fit_complete = False

    def fit(self, fit_data):
        self.imputer.fit(fit_data)
        self.scaler.fit(self.imputer.transform(fit_data))
        self.fit_complete = True

    def transform(self, transform_data):
        if not self.fit_complete:
            print("ERROR: Please, fit the data first.")
            return

        imputed_data = pd.DataFrame(self.imputer.transform(transform_data))
        data = self.scaler.transform(imputed_data)
        return pd.DataFrame(data, columns = transform_data.columns)