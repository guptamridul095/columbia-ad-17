## Setup

### Set up Python environment
Python 2 is currently required for Dataflow.
```
virtualenv venv --python=/usr/bin/python2.7
source ./venv/bin/activate
pip install -r requirements.txt
```
### Set up GCP credentials
```
gcloud auth login
gcloud auth application-default login
export GOOGLE_APPLICATION_CREDENTIALS=<PATH to GCS Key for gs://columbia-dl-storage-bucket>
```

## Preprocessing
### Set Constants
```
BUCKET=gs://[GCS Bucket]
NOW="$(date +%Y%m%d%H%M%S)"
OUTPUT_DIR="${BUCKET}/output_data/${NOW}"
```

### Run locally with Dataflow
```
python preprocessor/run_preprocessing.py \
--output_dir "${OUTPUT_DIR}"
```
### Run on the Cloud with Dataflow
```
python preprocessor/run_preprocessing.py --cloud \
--output_dir "${OUTPUT_DIR}"
```
  

## Training
### Set Constants
```
INPUT_DIR="${OUTPUT_DIR}"
MODEL_DIR="${BUCKET}/model/$(date +%Y%m%d%H%M%S)"
```

### Train locally with AI Platform
```
JOB_NAME="train_$(date +%Y%m%d%H%M%S)"

gcloud ai-platform local train \
--module-name trainer.task \
--package-path trainer \
--job-dir ${MODEL_DIR} \
-- \
--input-dir "${INPUT_DIR}"
```
