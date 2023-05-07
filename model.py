from sklearn.ensemble import GradientBoostingClassifier

class Model:
    def __init__(self):
        gbc = GradientBoostingClassifier()
        self.gbc = gbc
    def fit(self, X, y):
        self.gbc.fit(X, y)
        return

    def predict(self, X):
        return self.gbc.predict(X)

    def predict_proba(self, X):
        return self.gbc.predict_proba(X)

    def score(self, y1, y2):
        return self.gbc.score(y1, y2)
