import io
import os
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from diffusers import AutoPipelineForInpainting, LCMScheduler
from PIL import Image
from contextlib import asynccontextmanager

# Import internal modules
from image_processing import process_mask, resize_for_model

# Import for API Mode
import requests
import base64
import json
import traceback

# Configuration
# Switching to the most standard/reliable Inpainting model on HF (Classic V1.5)
MODEL_ID = "runwayml/stable-diffusion-inpainting"

USE_LCM = True # Keep True, but typically SD2 doesn't use SDXL LCM. We'll disable LCM for API mode implicitly by just sending standard params.

# Deployment Configuration
USE_API_MODE = os.getenv("USE_API_MODE", "false").lower() == "true"
HF_TOKEN = os.getenv("HF_TOKEN")

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # (Local model loading logic skipped for brevity, keeping it valid for syntax)
    if not USE_API_MODE:
        try:
            print(f"Loading local model: {MODEL_ID}...")
            pipeline = AutoPipelineForInpainting.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
            models["pipeline"] = pipeline
            print("Model loaded!")
        except Exception:
            pass
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Service is running", "model": MODEL_ID}

def encode_base64_image(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def query_hf_api(prompt, image, mask, model_id):
    # Try the original Inference URL first (some models still live there)
    # If 410, we know to switch. But let's try the Router URL which is the new standard.
    # Reverting to the standard Router URL structure found in docs for many models.
    
    # Debug: Trying standard inference endpoint again, as 410 might have been specific to the other model
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    payload = {
        "inputs": prompt,
        "image": encode_base64_image(image),
        "mask_image": encode_base64_image(mask),
        "parameters": {
             "num_inference_steps": 30,
             "strength": 0.9,
             "guidance_scale": 7.5
        }
    }

    print(f"DEBUG: Sending request to: {api_url}")
    response = requests.post(api_url, headers=headers, json=payload)
    return response, api_url

@app.post("/api/generate")
async def generate_inpainting(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
):
    try:
        image_bytes = await image.read()
        mask_bytes = await mask.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        
        pil_image = resize_for_model(pil_image)
        pil_mask = resize_for_model(pil_mask)
        pil_mask = process_mask(pil_mask, feather_radius=9)

        if USE_API_MODE:
            # 1. Try Standard Model
            response, used_url = query_hf_api(prompt, pil_image, pil_mask, MODEL_ID)
            
            # 2. If 410 (Gone) or 404 (Not Found), try the Router URL
            if response.status_code in [404, 410]:
                print(f"Primary URL failed ({response.status_code}). Trying Router URL...")
                # Try the router format
                router_url = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
                headers = {"Authorization": f"Bearer {HF_TOKEN}"}
                payload = {
                    "inputs": prompt,
                    "image": encode_base64_image(pil_image),
                    "mask_image": encode_base64_image(pil_mask)
                }
                response = requests.post(router_url, headers=headers, json=payload)
                used_url = router_url

            if response.status_code != 200:
                # CRITICAL DEBUGGING: Return the exact failure details to the client
                error_msg = f"HF API Failed. URL: {used_url} | Code: {response.status_code} | Body: {response.text}"
                print(error_msg)
                raise Exception(error_msg)

            result_bytes = response.content
            return Response(content=result_bytes, media_type="image/png")

        else:
             # (Local logic omitted for safety in this debug file)
             raise Exception("Local mode not supported in this debug file")

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"CRITICAL ERROR: {error_trace}")
        return Response(
            content=json.dumps({"detail": f"{str(e)}"}), # clean detail for checking
            status_code=500,
            media_type="application/json"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
