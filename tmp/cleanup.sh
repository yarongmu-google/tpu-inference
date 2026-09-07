#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: ./cleanup.sh <job-id>"
  exit 1
fi

JOB_ID=$1
# Replace <your-cdk-output-bucket> with your actual bucket URI
GCS_PATH="gs://<your-cdk-output-bucket>/$JOB_ID" 

echo "Creating local directories for Job $JOB_ID..."
# Using the JOB_ID in the local path prevents different runs from mixing
mkdir -p ./local_logs/$JOB_ID/client ./local_logs/$JOB_ID/server

echo "Downloading logs from $GCS_PATH..."
gcloud storage cp -r $GCS_PATH/client/* ./local_logs/$JOB_ID/client/ 2>/dev/null || true
gcloud storage cp -r $GCS_PATH/server/* ./local_logs/$JOB_ID/server/ 2>/dev/null || true

echo "Deleting remote GCS artifacts..."
gcloud storage rm -r $GCS_PATH

echo "Done! Logs saved to ./local_logs/$JOB_ID/"
