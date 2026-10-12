
import mlflow
from src.customer_churn.utils.ml_utils import load_model


class Prediction:

    def __init__(self, model_uri: str):
        self.model_uri = model_uri
        self.model = None
        self.model_version = None

    def _load_model(self):
        if self.model_uri.startswith("models:/") and "@" in self.model_uri:
            model_name, alias = self.model_uri[len("models:/"):].split("@", 1)

            client = mlflow.MlflowClient()
            version = client.get_model_version_by_alias(
                name=model_name,
                alias=alias,
            )

            # Pin inference to the exact version resolved above.
            self.model_version = str(version.version)
            exact_model_uri = f"models:/{model_name}/{self.model_version}"
        else:
            exact_model_uri = self.model_uri
            parts = exact_model_uri.rstrip("/").split("/")
            self.model_version = (
                parts[-1] if parts[-1].isdigit() else exact_model_uri
            )

        self.model = load_model(exact_model_uri)
        return self.model

    def predict(self, data):
        if self.model is None:
            self._load_model()

        prediction = self.model.predict(data)
        probability = self.model.predict_proba(data)

        return {
            "Prediction": int(prediction[0]),
            "Probability": float(probability[0][1]),
        }
