README.md
# Car Detection API

## Overview
This project provides a RESTful API that detects cars in an image, counts the number of red cars, and generates a textual description of the image.

## Features
- **Detects and counts cars** using YOLOv10.
- **Identifies red cars** based on color analysis.
- **Generates an image description** using BLIP.
- **FastAPI-based RESTful API**.

## Installation
### Prerequisites
Ensure you have **Python 3.8+** and [Poetry](https://python-poetry.org/docs/) installed.

### Setup
Clone the repository and install dependencies:
```sh
git clone https://github.com/ghazaros91/Car-Detection-API.git
cd car-detection-api

poetry install
```

## Usage
### Running the API
Start the FastAPI server:
```sh
poetry run python3 main.py
```
The API will be accessible at: `http://127.0.0.1:8000`

### API Endpoint
#### **POST /analyze-image/**
- **Headers:** `Content-Type: multipart/form-data`
- **Body:** Form-data with an image file under the key `image`
- **Response:**
```json
{
  "total_cars": 10,
  "red_cars": 3,
  "description": "The image depicts a busy urban street with several vehicles, including three trees, pedestrians, amidst tall buildings."
}
```

## Testing the API
To test the `/analyze-image/` endpoint, you can use the `test.py` script included in the project. This script will send an image to the API and print the results.

### Running the Test
```
python3 test.py
```
This will print the status code and the JSON response from the API, or display any errors encountered while parsing the response. The example response is below.
```json
{
  "total_cars": 23,
  "red_cars": 11,
  "description": "cars are parked in a large city street with a man standing in the middle"
}
```


## Approach
### 1. **Car Detection**
- Uses `YOLOv10` to detect and count cars.

### 2. **Red Car Identification**
- Extracts bounding box regions and analyzes color in **BGR** format.

### 3. **Image Description Generation**
- Uses `BLIP` to generate a text-based summary of the image.

## License
This project is licensed under the MIT License.

