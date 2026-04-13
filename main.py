from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.joblib")


class LungCancerInput(BaseModel):
    age: int
    gender: str
    bmi: float
    cholesterol_level: float
    hypertension: int
    asthma: int
    cirrhosis: int
    other_cancer: int
    family_history: str
    cancer_stage: str
    smoking_status: str
    treatment_type: str


@app.get("/")
def home():
    return {"message": "Lung cancer survival prediction API is running"}


@app.post("/predict")
def predict(data: LungCancerInput):
    gender_map = {
        "Male": 0,
        "Female": 1,
        "male": 0,
        "female": 1
    }

    family_history_map = {
        "Yes": 1,
        "No": 0,
        "yes": 1,
        "no": 0
    }

    cancer_stage_map = {
        "Stage I": 1,
        "Stage II": 2,
        "Stage III": 3,
        "Stage IV": 4,
        "stage i": 1,
        "stage ii": 2,
        "stage iii": 3,
        "stage iv": 4
    }

    smoking_status_map = {
        "Never Smoked": 0,
        "Former Smoker": 1,
        "Passive Smoker": 2,
        "Current Smoker": 3,
        "never smoked": 0,
        "former smoker": 1,
        "passive smoker": 2,
        "current smoker": 3
    }

    treatment_type_map = {
        "Surgery": 0,
        "Radiation": 1,
        "Chemotherapy": 2,
        "Combined": 3,
        "surgery": 0,
        "radiation": 1,
        "chemotherapy": 2,
        "combined": 3
    }

    gender_value = gender_map.get(data.gender)
    family_history_value = family_history_map.get(data.family_history)
    cancer_stage_value = cancer_stage_map.get(data.cancer_stage)
    smoking_status_value = smoking_status_map.get(data.smoking_status)
    treatment_type_value = treatment_type_map.get(data.treatment_type)

    if gender_value is None:
        return {"error": "gender must be Male or Female"}

    if family_history_value is None:
        return {"error": "family_history must be Yes or No"}

    if cancer_stage_value is None:
        return {"error": "cancer_stage must be Stage I, Stage II, Stage III, or Stage IV"}

    if smoking_status_value is None:
        return {"error": "smoking_status must be Never Smoked, Former Smoker, Passive Smoker, or Current Smoker"}

    if treatment_type_value is None:
        return {"error": "treatment_type must be Surgery, Radiation, Chemotherapy, or Combined"}

    features = np.array([[
        data.age,
        gender_value,
        data.bmi,
        data.cholesterol_level,
        data.hypertension,
        data.asthma,
        data.cirrhosis,
        data.other_cancer,
        family_history_value,
        cancer_stage_value,
        smoking_status_value,
        treatment_type_value
    ]])

    prediction = model.predict(features)[0]

    return {
        "prediction": int(prediction),
        "result": "Survived" if prediction == 1 else "Did not survive"
    }