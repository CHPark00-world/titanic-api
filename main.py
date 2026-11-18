from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('model.pkl')

class Passenger(BaseModel):
  Pclass: int
  Sex: int
  Age: float
  SibSp: int
  Parch: int
  Fare: float
  Embarked: int
  Title: int

@app.post('/predict')
def predict(p: Passenger):
  df = pd.DataFrame([p.model_dump()])
  result = model.predict(df)[0]
  return {'Survived': int(result)}

