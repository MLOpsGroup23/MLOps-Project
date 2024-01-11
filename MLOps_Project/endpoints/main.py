from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from MLOps_Project.predict_model import predict
from google.cloud import storage
import os

app = FastAPI()


# Downloading Model
# Initialise a client
print("Downloading Model")
cloud_id = os.environ["CLOUD_PROJECT_ID"]
storage_client = storage.Client(cloud_id)
# Create a bucket object for our bucket
bucket = storage_client.get_bucket('dtu-mlops-bucket1')
# Create a blob object from the filepath
blob = bucket.blob("LightningTrainedModel2.ckpt")
# Download the file to a destination
blob.download_to_filename("model2.ckpt")
print("Model downloaded and stored")
# Model downloaded and stored

# ===================== Routes =====================
@app.get("/", response_class=HTMLResponse)
async def homepage():
    return """
    <html>
        <head>
            <title>GR23 MLOps</title>
        </head>
        <body>
            <h1>Hello and welcome to our Machine Learning Operations project!</h1>
            <p>From this URL, you can access our model and try different stuff.</p>
        </body>
    </html>
    """

