from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(title="Mehfooz Musafir API", version="1.0")

MODEL_PATH = "weights/best_run4.pt"

print(f"Loading model from: {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None 

@app.get("/")
def home():
    return {"message": "Mehfooz Musafir API is Running! Send images to /detect"}

@app.post("/detect")
async def detect_helmets(file: UploadFile = File(...)):
    """
    Takes an uploaded image file, runs YOLOv8 detection, and returns the results.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    #(Bytes -> Image)
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    results = model.predict(image, conf=0.4)

    detections = []
    violation_count = 0

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            # Check for No Helmet (assuming class_name is 'no helmet' - check your data.yaml!)
            if class_name == "no helmet":
                violation_count += 1

            detections.append({
                "class": class_name,
                "confidence": round(confidence, 2),
                "box": box.xyxy[0].tolist() # Bounding box coordinates [x1, y1, x2, y2]
            })

    
    return {
        "filename": file.filename,
        "violation_count": violation_count,
        "is_safe": violation_count == 0,
        "all_detections": detections
    }