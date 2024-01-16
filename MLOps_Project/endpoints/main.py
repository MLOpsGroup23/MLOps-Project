from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from MLOps_Project.models.resnet import ResNet34
from MLOps_Project.predict_model import predict_single
from google.cloud import storage

from PIL import Image
import torch
import os
import io
import time
import pandas as pd

from MLOps_Project.endpoints.firestore import CustomFirestoreClient
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# ===================== Configs (should be moved somewhere else)  =====================
app = FastAPI()


# ===================== Configs (should be moved somewhere else)  =====================
# CLOUD_ID = os.environ["CLOUD_PROJECT_ID"]
CLOUD_ID = "pelagic-height-410710"
BUCKET_NAME = "dtu-mlops-bucket1"
DATABASE = "prediction-db"
COLLECTION = "predictions"

# ===================== Download Model  =====================
# Initialise a client
print("Downloading Model")
storage_client = storage.Client(CLOUD_ID)
# Create a bucket object for our bucket
bucket = storage_client.get_bucket(BUCKET_NAME)
# Create a blob object from the filepath
blob = bucket.blob("LightningTrainedModel2.ckpt")
# Download the file to a destination
blob.download_to_filename("model2.ckpt")
print("Model downloaded and stored")
# Model downloaded and stored
# Setup Model
print("Setting up the model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet34.load_from_checkpoint(checkpoint_path="./model2.ckpt")
model = model.to(device)


# ===================== Connect to Firebase  =====================

# Initialize a Firestore client
print("Making connection to Firestore")
db = CustomFirestoreClient(project=CLOUD_ID, database=DATABASE, collection=COLLECTION)


# ===================== Helper Functions  =====================


def bmp_to_tensor(bmp_data):
    # Load the image from bytes
    image = Image.open(io.BytesIO(bmp_data)).convert("L")

    # Resize the image to 28x28
    image = image.resize((28, 28))

    # Convert to a tensor
    tensor = torch.tensor(list(image.getdata()))
    tensor = tensor.view(1, 28, 28).float()

    # Normalize to fit dataset
    tensor = (tensor - torch.mean(tensor)) / torch.std(tensor)
    tensor = tensor.repeat_interleave(3, dim=0)  # Assumes a data channel requirement of 3

    return tensor

def get_processed_data_from_cloud(filename="test.pt"):
    print("Downloading reference data")
    # cloud_id = os.environ["CLOUD_PROJECT_ID"]
    cloud_id = "pelagic-height-410710"
    storage_client = storage.Client(cloud_id)
    # Create a bucket object for our bucket
    bucket = storage_client.get_bucket('dtu-mlops-bucket1')

    # Specify the folder name
    folder_name = 'data/processed/'

    # List all objects in the specified folder
    blobs = bucket.list_blobs(prefix=folder_name, delimiter='/')

    for blob in blobs:
        print(blob.name)
        if blob.name == folder_name + filename:
            print("Downloading file: {}".format(blob.name))
            blob.download_to_filename(f"datadrift/{filename}")
            print("File downloaded to {}".format(f"datadrift/{filename}"))

    # load saved blob 
    processed_data_pt = torch.load(f"datadrift/{filename}")
    return processed_data_pt    

def get_reference_data():
    # check if reference data already exists
    if os.path.exists('datadrift/reference_data.csv'):
        print("Reference data already exists")
        return pd.read_csv('datadrift/reference_data.csv')
    else: 
        # make datadrift folder if it does not exist
        if not os.path.exists('datadrift'):
            os.makedirs('datadrift')

        reference_data_pt = get_processed_data_from_cloud("test.pt")
        # convert to csv
        images = reference_data_pt[0]
        labels = reference_data_pt[1]

        # Make dataframe with labels
        data = {'label': labels}
        reference = pd.DataFrame(data)

        # save to csv file
        reference.to_csv('datadrift/reference_data.csv', index=False)
    
        return reference

# ===================== Routes =====================
@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_content = """
    <html>
        <head>
            <title>GR23 MLOps</title>
            <script>
                async function uploadImageAndPredict() {
                    var formData = new FormData();
                    formData.append('bmp_data', document.getElementById('imageInput').files[0]);

                    let response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    let result = await response.json();
                    let certainty = result.certainties[result.prediction];
                        certainty = (certainty*100).toString().substring(0, 5);

                    let anOrAnd = result.prediction[0] == "A" ? "an" : "a";

                    document.getElementById('predictionResult').innerText = "The model believes the image is " + anOrAnd + " " + result.prediction.toUpperCase() + " with a certainty of " + certainty + "%.";
                }
            </script>
        </head>
        <body>
            <h1>Hello and welcome to our Machine Learning Operations project!</h1>
            <p>From this URL, you can access our model and try different stuff.</p>

            <h3>Predict MNIST Fashion .bmp image here:</h3>
            <input type="file" id="imageInput" accept="image/bmp">
            <br/>

            <button onclick="uploadImageAndPredict()" style="margin-top: 20px;">Predict Image</button>
            <p id="predictionResult"></p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


def save_prediction(prediction: int, certainty: float):
    print("Saving prediction to Firestore")
    # Add prediction
    now = time.time()
    data = {'timestamp': now, 'uncertainty': certainty, 'label': prediction}
    db.add_prediction(data)


@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, bmp_data: UploadFile = File(...)):
    bmp_data_bytes = await bmp_data.read()
    tensor = bmp_to_tensor(bmp_data_bytes)

    pred = predict_single(model, tensor)

    ps = torch.exp(pred)
    _, top_class = ps.topk(1, dim=1)
    softmax = torch.nn.Softmax(dim=1)

    titles = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    certainties = softmax(pred)

    response = {
        "prediction": titles[top_class.item()],
        "certainties": {title: certainties[0][titles.index(title)].item() for title in titles},
    }

    print("Added background task for saving prediction to Firestore")
    background_tasks.add_task(save_prediction, top_class.item(), response["certainties"][response["prediction"]])

    return response


@app.get("/monitoring")
async def monitoring():

    titles = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    titles_dict = {i: titles[i] for i in range(len(titles))}

    # get current data 
    current = db.get_pd()
    # convert label column to int to title in titles
    current['label'] = current['label'].astype(int)
    current['label'] = current['label'].map(titles_dict)

    # get reference data
    reference = get_reference_data()
    # convert label column to int to title in titles
    reference['label'] = reference['label'].astype(int)
    reference['label'] = reference['label'].map(titles_dict)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html('report.html')

    # serve html file
    return FileResponse("report.html")
