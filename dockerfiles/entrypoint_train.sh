#!/bin/bash
set -e

# Fetch the API key from Google Cloud Secret Manager
WANDB_API_KEY=$(gcloud secrets versions access latest --secret="WAND_API")

# Log in to wandb
wandb login $WANDB_API_KEY

# Execute the main training script or command
exec "$@"
