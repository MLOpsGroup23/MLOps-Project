# Base image
FROM python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY MLOps_Project/ MLOps_Project/
COPY data/ data/

WORKDIR /

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# Set default values for environment variables
ENV SAVE_LOCATION="/models"
ENV N_EPOCHS=1

CMD python -u MLOps_Project/train_model.py --save_location $SAVE_LOCATION --n_epochs $N_EPOCHS
