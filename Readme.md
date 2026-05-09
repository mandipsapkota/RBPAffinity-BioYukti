#  Overview

The dataset and final model files for this project exceed GitHub’s file size limits. You can download all necessary assets from this [Google Drive folder](https://drive.google.com/drive/folders/1YBwlrvNZaNffMNS3tfqObjw4N9bJSTeF?usp=sharing).

## Introduction

This project predicts the affinity and strength of affinity of RNA binding proteins, especially the TPD43 protein, by leveraging transformer-based architectures and genomic data processing.

## Model Architecture

The system uses a **DistilBERT** backbone adapted for genomic sequences, featuring both regression and classification heads to predict binding affinity and site presence.

## Data Preparation

The pipeline processes genomic data using standard bioinformatic formats:

* **.bed files:** Used to locate confirmed protein binding sites.
* **.fa files:** Used to extract raw nucleotide sequences.
* **.gtf files:** Used to define genomic regions for the negative dataset.
* **Sampling:** Positive samples are drawn from exons, while negative samples are drawn from introns at a 3:1 ratio.

## Setup and Installation
You can either see how we trained the model, or use our app using a model that we already trained. 

### 1. Model Training

To set up the environment for training or local experimentation:

1. Create a virtual environment:
`python -m venv venv`
2. Activate the environment:
* **Windows:** `.\venv\Scripts\activate`
* **macOS/Linux:** `source venv/bin/activate`


3. Install dependencies:
```bash
pip install -r requirements.txt

```
4. Run the notebook present in notebooks/rbp-prediction-pipeline-tdp43.ipynb. The required file can be found in google drive mentioned at the top of this readme.



### 2. Running the Application

To launch the API/Web interface:

1. Create and activate a separate virtual environment.
2. Place the downloaded model file into backend/idp43_2.0/model_weights.pth.
3. Install the application dependencies:

```bash
    pip install -r requirements.txt
```
4.  Start the server:
    
```bash
    uvicorn main:app --reload --port 8000
```
5. Open the `index.html` file located in frontend folder.

With this much being done, you are ready to go with the application.