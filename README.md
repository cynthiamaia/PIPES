# PIPES API

This is an API built with FastAPI that allows you to retrieve PIPES metadata. The API provides endpoints to fetch datasets and retrieve algorithm-specific data.

## Features

- Retrieves data from files based on the dataset ID and algorithm name.

- Automatically handles missing and infinite values ​​in the dataset.

- Returns clean JSON responses.

##Requirements

- Python 3.8 or higher
- FastAPI
- Pandas
- NumPy
- Uvicorn (to run the server)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up the `datasets` directory: Make sure a folder named `datasets` exists in the root directory. Within this folder, organize subfolders by dataset ID, each containing CSV files.

Example structure:

```
datasets/
1/
algorithm1_data.csv
algorithm2_data.csv
2/
algorithm1_data.csv
```

## Usage

1. Run the API server:

```bash
uvicorn main:app --reload
```

2. Access the API documentation:

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- URL SWAGGER UI: [https://pipes-production.up.railway.app/docs](https://pipes-production.up.railway.app/docs)

## Endpoints

### **Retrieve algorithm data**

- **URL:** `/data/{dataset_id}/{algorithm_name}`
- **Method:** `GET`
- **Description:** Fetches data from a CSV file that matches the given dataset ID and algorithm name.

#### **Path parameters:**

| Parameter | Type | Description |
| ---------------- | ----- | ---------------------------------------- |
| `dataset_id` | `int` | The ID of the dataset folder. |
| `algorithm_name` | `str` | The name of the algorithm to search for. |

#### **Responses:**

| Status code | Description | Example |
| ----------- | ----------------------------- | -------------------------------------------------- ------ |
| `200` | Success, returns the dataset. | `[ {"column1": "value1", "column2": "value2"} ]` |
| `404` | Dataset or file not found. | `{ "detail": "Dataset ID 1 not found." }` |
| `500` | Error processing file. | `{ "detail": "Error reading file: file is corrupt." }` |

#### **Example request:**

```bash
GET /data/1/algorithm1
```

#### **Example response:**

```json
[
{
"column1": "value1",
"column2": "value2"
},
{
"column1": "value3",
"column2": "value4"
}
]
```

## Under construction

Add POST endpoint to send data