from fastapi import FastAPI, Request
from MLOps_Project.predict_model import predict

app = FastAPI()

# ===================== Routes =====================
@app.get("/")
def read_root(request: Request):
    return {"Hello": "World"}   

@app.get("/test_dst")
def dst_test():
    return {
        "Message": "This is a test message - please ignore me."
    }

@app.get("/hello_world")
def hello_server():
    return {
        "Message": "Hello from an updated server"
    }

@app.get("/hello_london")
def hello_server():
    return {
        "Message": "Hello from London! Bing Bong."
    }