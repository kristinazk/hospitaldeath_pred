import argparse
import pandas as pd
from model import Model
from preprocessor import Preprocessor
import json
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required = True)
parser.add_argument('--inference', required = False)
args = parser.parse_args()

class Pipeline:
    def __init__(self,):
        self.model = Model()
        self.preprocessor = Preprocessor()

    def run(self, X, test = False):

        X = pd.read_csv(X)

        if test:

            loaded_model = pickle.load(open('fitted_model.sav', 'rb'))
            loaded_preprocessor = pickle.load(open('fitted_preprocessor.sav', 'rb'))

            X_test = loaded_preprocessor.transform(X)
            proba = loaded_model.predict_proba(X_test)

            dictionary = {
                "predict_probas": proba.tolist(),
                "threshold": 0.7,
            }

            json_object = json.dumps(dictionary, indent = 2)

            with open("predictions.json", "w") as outfile:
                outfile.write(json_object)

        else:
            X_train = X[X.columns.difference(['In-hospital_death'])]
            y_train = X['In-hospital_death']
            preprocessor = self.preprocessor
            preprocessor.fit(X_train)
            X_train = preprocessor.transform(X_train)

            pickle.dump(preprocessor, open('fitted_preprocessor.sav', 'wb'))

            model = self.model
            model.fit(X_train, y_train)

            pickle.dump(model, open('fitted_model.sav', 'wb'))

pipeline = Pipeline()
pipeline.run(args.data_path, args.inference)