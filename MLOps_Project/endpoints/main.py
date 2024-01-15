from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from MLOps_Project.models.model import ResNet34
from MLOps_Project.predict_model import predict_single
from google.cloud import storage

from PIL import Image
import torch
import os
import io

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
# Setup Model
print("Setting up the model")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet34.load_from_checkpoint(checkpoint_path="./model2.ckpt")
model = model.to(device)


# ===================== Helper Functions  =====================

def bmp_to_tensor(bmp_data):
     # Load the image from bytes
    image = Image.open(io.BytesIO(bmp_data)).convert('L')

    # Resize the image to 28x28
    image = image.resize((28, 28))

    # Convert to a tensor
    tensor = torch.tensor(list(image.getdata()))
    tensor = tensor.view(1, 28, 28).float()  

    # Normalize to fit dataset
    tensor = (tensor - torch.mean(tensor))/torch.std(tensor)

    return tensor

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


@app.post("/predict")
async def predict(bmp_data: UploadFile = File(...)):
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
        "certainties": {title: certainties[0][titles.index(title)].item() for title in titles}
    }
    return response