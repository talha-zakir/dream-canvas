import io
import os
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from diffusers import AutoPipelineForInpainting, LCMScheduler
from PIL import Image
from contextlib import asynccontextmanager
from huggingface_hub import InferenceClient

# Import internal modules
from image_processing import process_mask, resize_for_model

# Configuration
# MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev" # Uncomment for FLUX (Requires High VRAM)
MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
USE_LCM = True # Set to True for fast generation (2-4s), False for higher quality (30s)

# Deployment Configuration
USE_API_MODE = os.getenv("USE_API_MODE", "false").lower() == "true"
HF_TOKEN = os.getenv("HF_TOKEN")

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model on startup to avoid re-loading on every request.
    If USE_API_MODE is True, we skip local model loading.
    """
    if USE_API_MODE:
        print("Starting in API Mode (Serverless Inference). Skipping local model load.")
        if not HF_TOKEN:
             print("WARNING: HF_TOKEN is not set. API calls might be rate limited or fail.")
    else:
        print(f"Loading local model: {MODEL_ID}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        try:
            pipeline = AutoPipelineForInpainting.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
                use_safetensors=True
            )
            
            # Optimization: Enable CPU offload to save VRAM
            if device == "cuda":
                pipeline.enable_model_cpu_offload()
            
            # LCM (Latent Consistency Model) extraction for Speed
            if USE_LCM and "stable-diffusion-xl" in MODEL_ID:
                print("Applying LCM LoRA for speed...")
                pipeline.load_lora_weights("latent-consistency/lcm-lora-sdxl")
                pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

            models["pipeline"] = pipeline
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    yield
    
    # Cleanup
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    mode = "API Inference" if USE_API_MODE else "Local Inference"
    return {"status": "Service is running", "model": MODEL_ID, "mode": mode}

@app.post("/api/generate")
async def generate_inpainting(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
):
    try:
        # Read images
        image_bytes = await image.read()
        mask_bytes = await mask.read()

        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_mask = Image.open(io.BytesIO(mask_bytes)).convert("L") # Mask should be grayscale

        # Preprocessing
        # 1. Resize to 1024x1024 (standard for SDXL/Flux)
        pil_image = resize_for_model(pil_image)
        pil_mask = resize_for_model(pil_mask)

        # 2. Feather the mask to blend edges
        pil_mask = process_mask(pil_mask, feather_radius=9)

        if USE_API_MODE:
            # Use Hugging Face Inference API
            client = InferenceClient(token=HF_TOKEN)
            
            # Call the specific SDXL Inpainting model on HF
            try:
                result_image = client.image_to_image(
                    model=MODEL_ID,
                    prompt=prompt,
                    image=pil_image, 
                    mask_image=pil_mask, 
                    parameters={"strength": 0.99, "num_inference_steps": 25}
                )
            except Exception as hf_error:
                print(f"Hugging Face API Error: {hf_error}")
                # Re-raise with a clear message that will be sent to frontend
                raise HTTPException(status_code=500, detail=f"Hugging Face API Error: {str(hf_error)}")
            # result_image is a PIL Image
            result = result_image

        else:
            # Local Inference
            if "pipeline" not in models:
                 raise HTTPException(status_code=500, detail="Model not loaded")
            
            pipeline = models["pipeline"]
            
            # Adjust steps based on LCM usage
            num_inference_steps = 4 if USE_LCM else 30
            guidance_scale = 1.0 if USE_LCM else 7.5
    
            generator = torch.Generator(device="cpu").manual_seed(42) # For reproducibility
    
            result = pipeline(
                prompt=prompt,
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                strength=0.99, # High strength to ensure inpainting fills the area
            ).images[0]

        # Return the generated image
        output_io = io.BytesIO()
        result.save(output_io, format="PNG")
        output_io.seek(0)

        return Response(content=output_io.getvalue(), media_type="image/png")

    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
