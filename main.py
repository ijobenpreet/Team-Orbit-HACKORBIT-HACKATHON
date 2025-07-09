from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import socketio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CyberShield Social Media Sentinel API")
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app.mount("/socket.io", socketio.ASGIApp(sio))

try:
    tokenizer = DistilBertTokenizer.from_pretrained("./cyber_shield_model")
    model = DistilBertForSequenceClassification.from_pretrained("./cyber_shield_model")
    model.eval()
    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model or tokenizer: {e}")
    raise e

class TextInput(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "CyberShield Social Media Sentinel API. Use /predict for predictions or /docs for Swagger UI."}

def predict_text(text: str):
    if not text.strip():
        logger.warning("Empty input text received.")
        return {"error": "Input text cannot be empty."}
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).numpy()[0]
        labels = ["hate_speech", "cyberbullying", "fake_account", "incitement_violence", "threat_safety"]
        return {label: float(prob) for label, prob in zip(labels, probs)}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}

@app.post("/predict")
async def predict(input_data: TextInput):
    prediction = predict_text(input_data.text)
    if "error" in prediction:
        raise HTTPException(status_code=400, detail=prediction["error"])
    await sio.emit("new_alert", prediction)
    logger.info(f"Prediction sent: {prediction}")
    return {"prediction": prediction}

@sio.event
async def connect(sid, environ):
    logger.info(f"WebSocket connected: {sid}")

@sio.event
async def disconnect(sid):
    logger.info(f"WebSocket disconnected: {sid}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)