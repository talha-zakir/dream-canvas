# Deployment Guide: Free Hugging Face Spaces & Vercel

This guide explains how to deploy your "Pencil-to-Inpaint" app for free. Since AI models (like SDXL) require powerful GPUs, and standard free tiers only offer CPUs, we will use a hybrid approach or the "Serverless API" mode.

## Architecture

1.  **Backend (Hugging Face Space)**: hosts the FastAPI logic using Docker.
2.  **Frontend (Vercel)**: hosts the Next.js React app.
3.  **Inference**:
    *   **Option A (Free CPU)**: The backend calls Hugging Face's *Serverless Inference API* instead of running the model locally.
    *   **Option B (Paid GPU)**: You upgrade the Space to a GPU, and the backend runs the model locally as originally designed.

---

## Part 1: Backend Deployment (Hugging Face Spaces)

1.  **Create a Space**:
    *   Go to [Hugging Face Spaces](https://huggingface.co/spaces).
    *   Click **Create new Space**.
    *   **Name**: `pencil-inpaint-backend`.
    *   **SDK**: Select **Docker**.
    *   **Hardware**: `CPU Basic (Free)`.
    *   Click **Create Space**.

2.  **Get a Hugging Face Token**:
    *   Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens).
    *   Create a new token (Role: `read`).
    *   Copy it.

3.  **Configure Environment Variables**:
    *   In your new Space, go to **Settings**.
    *   Scroll to **Variables and secrets**.
    *   Add Variable: `USE_API_MODE` = `true` (This enables the API-only mode for free CPU usage).
    *   Add Secret: `HF_TOKEN` = `your_token_here` (Paste your token).

4.  **Upload Code**:
    *   You can clone the Space repo and copy your `backend/` files into it, OR upload via the web interface.
    *   **Important**: The `Dockerfile` must be in the root of the Space repository.
    *   Copy everything inside your local `backend/` folder to the **root** of the Space repository.
    *   Files needed: `Dockerfile`, `main.py`, `requirements.txt`, `image_processing.py`.

5.  **Build**:
    *   HF Spaces will automatically build the Docker container. Wait for it to show "Running".
    *   Note the "Direct URL" (top right menu > Embed this space > Direct URL). It looks like: `https://username-space-name.hf.space`.

---

## Part 2: Frontend Deployment (Vercel)

1.  **Push to GitHub**:
    *   Push your entire project (or just the `frontend` folder) to a GitHub repository.

2.  **Deploy on Vercel**:
    *   Go to [Vercel](https://vercel.com) and click **Add New > Project**.
    *   Import your GitHub repository.
    *   **Root Directory**: Edit this to select `frontend` (if you uploaded the whole project).

3.  **Configure Environment Variables**:
    *   In the Vercel deployment screen (before clicking Deploy), expand **Environment Variables**.
    *   Add `NEXT_PUBLIC_API_URL`.
    *   Value: The **Direct URL** of your Hugging Face Space (e.g., `https://username-space-name.hf.space`).
    *   *Note: Do not add a trailing slash.*

4.  **Deploy**:
    *   Click **Deploy**.
    *   Vercel will build and give you a URL (e.g., `https://pencil-inpaint.vercel.app`).

---

## Part 3: Verify & Test

1.  Open your Vercel URL.
2.  Upload an image and draw a mask.
3.  Click Generate.
4.  **Troubleshooting**:
    *   If it fails, check the **Vercel Logs** (Function logs) and the **Hugging Face Space Logs**.
    *   If you see "CORS" errors, ensure your Backend `main.py` has `allow_origins=["*"]` (It does in the provided code).
    *   If the image generation is bad or fails with 503, the generic Hugging Face Free API might be overloaded. You may need to retry or switch models in `main.py`.

## Why this approach?

Running `SDXL` locally on the free 2-vCPU Hugging Face Space will either **crash** (Out of Memory) or take **2-5 minutes per image**. By setting `USE_API_MODE=true` and `HF_TOKEN`, the backend merely acts as a secure proxy, offloading the heavy math to Hugging Face's global serverless cluster, which is much faster for testing.
