from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

class Candidate(BaseModel):
    experience_years: float
    technical_score: float
    

class HiringAPI:
    def __init__(self):
        self.app = FastAPI(title="Hiring Prediction API")
        self.model = joblib.load("./pkl_files/model.pkl")
        self.scaler = joblib.load("./pkl_files/scaler.pkl")
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/")
        def welcome():
            return {"message": "Welcome to the Hiring Prediction API!"}

        @self.app.post("/predict")
        def predict(candidate: Candidate):
            try:
                input_df = pd.DataFrame(
                    [[candidate.experience_years, candidate.technical_score]],
                    columns=["experienced_years", "technical_score"]
                )
                input_scaled = self.scaler.transform(input_df)
                prediction = self.model.predict(input_scaled)
                result = "Hired" if prediction[0] == 0 else " Not Hired"
                return {"prediction": result}
            except Exception as e:
                return {f"Error": {e}}

    def get_app(self):
        return self.app