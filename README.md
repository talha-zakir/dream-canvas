# Pencil-to-Inpaint Prototype

A "Pencil-to-Inpaint" web application where you can upload an image, mask an area with a pencil tool, and use Generative AI to fill it comfortably.

## Project Structure

- **backend/**: FastAPI server handling the AI model (SDXL Inpainting / Flux.1-Fill).
  - `main.py`: Entry point, loads model, handles API requests.
  - `image_processing.py`: Handles mask feathering (Gaussian Blur) and resizing.
- **frontend/**: Next.js application with Fabric.js canvas.
  - `components/InpaintingCanvas.tsx`: The interactive canvas for drawing masks.
  - `app/page.tsx`: Main UI.

## Features Implemented

1.  **Feathering**: The backend applies a Gaussian Blur to the uploaded mask (`feather_radius=9`). this ensures that the in-painted area blends smoothly with the original image, avoiding harsh, pixelated edges.
2.  **Latent Consistency (LCM)**: The FastAPI server is configured to try loading `lcm-lora-sdxl`. This reduces inference steps from ~30-50 to just 4-8, speeding up generation from ~30s to ~3s on supported GPUs.
3.  **VRAM Optimization**: The model uses `enable_model_cpu_offload()` to offload components to CPU when not in use, allowing SDXL to run on 8GB+ VRAM cards (instead of 12GB+).

## Setup Instructions

### 1. Backend

Prerequisites: Python 3.10+ and a GPU (NVIDIA recommended).

```bash
cd backend
pip install -r requirements.txt
python main.py
```
The server will start at `http://localhost:8000`. On first run, it will download several GBs of model weights.

### 2. Frontend

Prerequisites: Node.js 18+.

```bash
cd frontend
npm install # (If not already installed)
npm run dev
```
Open `http://localhost:3000` in your browser.

## How to Use

1.  Upload an image using the "Upload Image" button.
2.  Use the mouse to draw (white pencil) over the object you want to change.
3.  Adjust brush size with the slider if needed.
4.  Type a prompt (e.g., "a red sports car", "a pile of gold coins").
5.  Click **Generate**.
6.  Wait for the AI to dream (check the backend console for progress).

## Technical Implementation Details

- **Mask Handling**: The frontend exports two PNGs: the original image and a binary mask (black background, white drawing).
- **Backend Processing**: `image_processing.process_mask` applies `cv2.GaussianBlur` to the binary mask. This gradients the edges (gray values), telling the AI to partially preserve the original pixels at the border, creating a seamless transition.
- **Model**: Default is `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`. You can switch to Flux.1 in `main.py` by uncommenting the `MODEL_ID` line (requires HuggingFace login).
