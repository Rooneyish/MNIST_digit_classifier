from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from PIL import Image
import numpy as np
import io
import asyncio
from src.model import Artificial_Neural_Network

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App Starting: loading MNIST model...")
    ann = Artificial_Neural_Network(
        input_size=28*28, hidden1_size=128, hidden2_size=64, output_size=10
    )
    ann.load("model.pkl")  # load saved weights
    models['mnist_ann'] = ann
    yield
    print("Shutting Down App...")

app = FastAPI(lifespan = lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def preprocess_image(img: Image.Image):
    img = img.convert("L")
    img = img.resize((28,28))
    img_arr = np.array(img)
    img_arr = 255-img_arr
    img_arr = img_arr/255.0
    img_arr = img_arr.flatten().reshape(1,-1)
    return img_arr

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        img = Image.open(io.BytesIO(file_bytes))
        img_arr = preprocess_image(img)
        model = models.get("mnist_ann")

        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(None, lambda: model.forward_prop(img_arr))
        digit = int(np.argmax(prediction, axis=1)[0])

        return{"status_code": 200, "predicted_digit": digit}
    
    except Exception as e:
        return JSONResponse(status_code = 500, content = {"status_code": 500, "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)