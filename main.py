from fastapi import FastAPI, HTTPException
from typing import List
import os
import pandas as pd
import numpy as np

app = FastAPI(
    title="PIPES",
    description="This is a sample API using FastAPI",
    version="1.0.0",
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

BASE_DIR = "datasets"

@app.get("/")
async def hello_world():
    return {"message": "Hello, World!"}


# @app.get("/data/{dataset_id}/{algorithm_name}")
@app.get(
    "/data/{dataset_id}/{algorithm_name}",
    tags=["Data Retrieval"],
    summary="Retrieve algorithm data from a dataset",
    description=(
        "Fetches the data from a CSV file that matches the provided dataset ID and algorithm name. "
        "If no matching file is found or the dataset ID does not exist, an error is returned."
    ),
    responses={
        200: {
            "description": "A list of records from the matching CSV file.",
            "content": {
                "application/json": {
                    "example": [
                        {"column1": "value1", "column2": "value2"},
                        {"column1": "value3", "column2": "value4"},
                    ]
                }
            },
        },
        404: {
            "description": "Dataset or file not found.",
            "content": {
                "application/json": {
                    "example": {"detail": "Dataset ID 1 not found."}
                }
            },
        },
        500: {
            "description": "Error processing the file.",
            "content": {
                "application/json": {
                    "example": {"detail": "Error reading file: File is corrupted."}
                }
            },
        },
    },
)
async def get_algorithm_data(dataset_id: int, algorithm_name: str):
    dataset_dir = os.path.join(BASE_DIR, str(dataset_id))
    
    if not os.path.isdir(dataset_dir):
        raise HTTPException(status_code=404, detail=f"Dataset ID {dataset_id} not found.")
    
    files = [
        f for f in os.listdir(dataset_dir)
        if algorithm_name in f and f.endswith(".csv")
    ]
    
    if not files:
        raise HTTPException(status_code=404, detail=f"No files found for algorithm {algorithm_name}.")
    
    file_path = os.path.join(dataset_dir, files[0])
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
    
    df = df.replace([float('inf'), float('-inf')], None).where(pd.notnull(df), None)
    df = df.drop(columns=["Using_imputer", "Using_cat"])
    # df = df.fillna(value=None) 
    for column in df.select_dtypes(include=[np.float64]).columns:
        df[column] = df[column].astype(str)
    
    return df.to_dict(orient="records")
