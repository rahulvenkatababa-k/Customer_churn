from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.customer_churn.components.prediction import Prediction
from prometheus_fastapi_instrumentator import Instrumentator

from fastapi import FastAPI, BackgroundTasks
from src.customer_churn.components.prediction_logger import save_prediction

app = FastAPI(
    title="Customer Churn Prediction API 🛒",
    description="API for predicting Customer Churn",
    version="1.0.0"
)
Instrumentator().instrument(app=app).expose(app=app, endpoint="/metrics")
@app.get("/")
def root():
    return {
        "message" : "Customer Churn prediction API is running 🛒"
    }
@app.get("/health")
def health():
    return {
        "Status" : "Healthy ✨"
    }

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

MODEL_URI = "models:/customer_churn_model@champion"
predictor = Prediction(model_uri=MODEL_URI)

@app.post("/predict")
def predict_customer(
    customer: CustomerData,
    background_tasks: BackgroundTasks,
):
    data = pd.DataFrame([customer.model_dump()])

    result = predictor.predict(data=data)
    model_version = predictor.model_version

    background_tasks.add_task(
        save_prediction,
        features=customer.model_dump(),
        prediction=result["Prediction"],
        probability=result["Probability"],
        model_version=model_version,
    )

    return {
    "Prediction": result["Prediction"],
    "Probability": result["Probability"],
}