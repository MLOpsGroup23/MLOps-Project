from fastapi import FastAPI, Request
from MLOps_Project.predict_model import predict

app = FastAPI()

# ===================== Routes =====================
@app.get("/")
def read_root(request: Request):
    return {"Hello": "World"}   

