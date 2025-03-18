# FastAPI Integration
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle as pickle

app = FastAPI()

# Load the trained model
with open("salary_regressor.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Define request body model
class SalaryInput(BaseModel):
    age: int
    experience: int
    education: int

@app.post("/predict_salary")
def predict_salary(data: SalaryInput):
    input_features = np.array([[data.age, data.education, data.experience]])
    predicted_salary = loaded_model.predict(input_features)
    return {"predicted_salary": predicted_salary[0]}
