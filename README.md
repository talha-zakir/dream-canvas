# Dream Canvas: AI Inpainting

A "Pencil-to-Inpaint" web application where you can upload an image, mask an area with a pencil tool, and use Generative AI (SDXL) to fill it.

![Demo](https://your-demo-image-url.com/placeholder.png)

## Repository Structure

This is a monorepo containing both the frontend and backend code.

- **`/frontend`**: The Next.js (React) application. Deployed on **Vercel**.
- **`/backend`**: The FastAPI (Python) server logic.
- **`/hf_space_deployment`**: A flattened version of the backend ready for **Hugging Face Spaces**.

## Deployment

### 1. Backend (Hugging Face Spaces)
The backend runs on Hugging Face Spaces using Docker.
- **Automation**: Setup via GitHub Actions (`.github/workflows/deploy_to_hf.yml`). Any change to `/hf_space_deployment` is automatically pushed to the Space.
- **Manual Setup (One-time)**:
    - Add `HF_TOKEN` to GitHub Repository Secrets.
- **SDK**: Docker
- **Hardware**: CPU (Free Tier supported via API Mode)
- **Environment Variables**:
    - `USE_API_MODE`: `true` (Enables serverless inference proxy)
    - `HF_TOKEN`: `hf_...` (Your Hugging Face Read Token)

### 2. Frontend (Vercel)
The frontend runs on Vercel.
- **Root Directory**: `frontend`
- **Environment Variables**:
    - `NEXT_PUBLIC_API_URL`: `https://your-space-name.hf.space`

## Development

### Prerequisites
- Node.js 18+
- Python 3.10+

### Local Setup
1.  **Backend**:
    ```bash
    cd backend
    pip install -r requirements.txt
    python main.py
    ```
2.  **Frontend**:
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

## License
MIT
