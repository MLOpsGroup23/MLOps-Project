# Base image
FROM python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt-get install ffmpeg libsm6 libxext6  -y 

COPY requirements_fastapi_predict.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY MLOps_Project/ MLOps_Project/
# copy predict_model.py
# COPY MLOps_Project/predict_model.py MLOps_Project/predict_model.py
COPY models/ models/

# Remove predict_model.py from MLOps_Project
# RUN rm MLOps_Project/predict_model.py
# RUN rm MLOps_Project/train_model.py


WORKDIR /
RUN pip install --upgrade -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# Run FastAPI app
CMD ["uvicorn", "MLOps_Project.endpoints.main:app", "--host", "0.0.0.0", "--port", "80"]
