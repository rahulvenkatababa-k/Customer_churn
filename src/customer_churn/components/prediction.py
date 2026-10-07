import mlflow
from src.customer_churn.utils.ml_utils import load_model

class Prediction:

    def __init__(self, model_uri: str):
        self.model_uri = model_uri
        self.model = None

    def _load_model(self):
        self.model = load_model(self.model_uri)
        return self.model

    def predict(self, data):
        if self.model is None:
            self._load_model()
        prediction = self.model.predict(data)
        probability = self.model.predict_proba(data)

        return {
            "Prediction" : int(prediction[0]),
            "Probability" : float(probability[0][1])
        }