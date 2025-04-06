from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from io import BytesIO
import torch

class CarsAnalyzer:
    def __init__(self):
        try:
            self.car_detector = YOLO("yolov10n.pt")
            self.image_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=True)
            self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

    def detect_red_cars(self, image_data: np.ndarray, bounding_boxes: list):
        try:
            detected_red_cars = 0
            for bbox in bounding_boxes:
                x1, y1, x2, y2 = map(int, bbox[:4])
                region_of_interest = image_data[y1:y2, x1:x2]
                avg_color_value = np.mean(region_of_interest, axis=(0, 1))
                if avg_color_value[2] > 120 and avg_color_value[2] > avg_color_value[1] and avg_color_value[2] > avg_color_value[0]:
                    detected_red_cars += 1
            return detected_red_cars
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error detecting red cars: {str(e)}")

    def generate_image_caption(self, image_bytes: bytes) -> str:
        try:
            pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
            model_input = self.image_processor(pil_img, return_tensors="pt")
            with torch.no_grad():
                generated_caption = self.caption_model.generate(**model_input)
            caption = self.image_processor.batch_decode(generated_caption, skip_special_tokens=True)[0]
            return caption
        except Exception as e:            
            raise HTTPException(status_code=500, detail=f"Error generating caption: {str(e)}")

    def detect_cars(self, image_data: np.ndarray):
        try:
            detection_results = self.car_detector(image_data)
            car_count = 0
            car_bboxes = []
            for result in detection_results:
                for bbox in result.boxes.data:
                    if int(bbox[5]) == 2:  # Class 2 is car
                        car_count += 1
                        car_bboxes.append(bbox)
            return car_count, car_bboxes
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error detecting cars from results: {str(e)}")

    async def analyze_cars(self, image: UploadFile):
        try:
            image_content = await image.read()
            img_array = np.frombuffer(image_content, np.uint8)
            img_cv2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            car_count, car_bboxes = self.detect_cars(img_cv2)
            red_car_count = self.detect_red_cars(img_cv2, car_bboxes)
            caption = self.generate_image_caption(image_content)

            return {
                "total_cars": car_count,
                "red_cars": red_car_count,
                "description": caption
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error analyzing image: {str(e)}")


app = FastAPI()
cars_analyzer = CarsAnalyzer()

@app.post("/analyze-image/")
async def analyze_image(image: UploadFile = File(...)):
    try:
        return await cars_analyzer.analyze_cars(image)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
