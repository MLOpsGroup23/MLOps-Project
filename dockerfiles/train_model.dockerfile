# Base image
FROM python:3.10-slim

ARG PROJECT_ID

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY MLOps_Project/ MLOps_Project/
COPY data/ data/
COPY entrypoint.sh entrypoint.sh

WORKDIR /

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# Set default values for environment variables
ENV SAVE_LOCATION="/models"
ENV N_EPOCHS=1

# Set gcloud project
RUN gcloud config set project ${PROJECT_ID}

ENTRYPOINT ["/entrypoint.sh"]
CMD python -u MLOps_Project/train_model.py --save_location $SAVE_LOCATION --n_epochs $N_EPOCHS
