from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Loan Default Prediction API")

try:
    model = joblib.load("best_model.pkl")
except Exception:
    model = None

class LoanInput(BaseModel):
    Credit_Score: int = Field(..., example=758)
    LTV: float = Field(..., example=98.7)
    dtir1: float = Field(..., example=45.0)

    loan_type: str = Field(..., example="type1")
    age: str = Field(..., example="25-34")
    Region: str = Field(..., example="south")

class PredictRequest(BaseModel):
    input: LoanInput

@app.get("/")
def health_check():
    return {"message": "API Running"}

@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found.")

    try:
        input_dict = request.input.dict()
        input_df = pd.DataFrame([input_dict])

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df).max(axis=1)[0])

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "status": "Success"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction Error: {str(e)}")