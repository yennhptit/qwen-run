import io
import os
import uuid
from datetime import datetime
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from urllib.parse import urlparse
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
from diffusers.utils import load_image
from PIL import Image
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import uvicorn

# ================== Load Cloudinary config ==================
load_dotenv()
CLOUD_NAME = os.getenv("CLOUD_NAME")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=API_KEY,
    api_secret=API_SECRET,
)

# ================== FastAPI setup ==================
app = FastAPI(title="Qwen Image Edit CPU Only")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image_url: str
    prompt: str

# ================== ThreadPool & task tracking ==================
MAX_WORKERS = 2
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
task_status: Dict[str, Dict] = {}  # task_id -> {"progress": float, "image_url": str}

# CPU device
device = "cpu"

# ================== Model setup ==================
model_id = "Qwen/Qwen-Image-Edit"
torch_dtype = torch.bfloat16

# Transformer quantization
transformer_quant = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["transformer_blocks.0.img_mod"]
)
transformer = QwenImageTransformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=transformer_quant,
    torch_dtype=torch_dtype
).to(device)

# Text encoder quantization
text_encoder_quant = TransformersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    subfolder="text_encoder",
    quantization_config=text_encoder_quant,
    torch_dtype=torch_dtype
).to(device)

# Pipeline
pipe = QwenImageEditPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch_dtype
)
pipe.load_lora_weights("flymy_qwen_image_edit_inscene_lora.safetensors")
pipe.enable_model_cpu_offload()

generator = torch.Generator(device=device).manual_seed(42)

# ================== Image generation & upload ==================
def generate_and_upload(task_id: str, image_url: str, prompt: str) -> str:
    """
    Generate image and upload to Cloudinary.
    Updates task_status[task_id]["progress"] for fake progress.
    """
    try:
        # Fake callback function
        def callback(percent: float):
            task_status[task_id]["progress"] = percent

        # Start progress
        callback(10.0)

        # Load input image
        image = load_image(image_url)

        # Run pipeline (no callback argument in new version)
        edited_image = pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=32,
            generator=generator
        ).images[0]

        # Progress done
        callback(80.0)

        # Save to buffer
        buf = io.BytesIO()
        edited_image.save(buf, format="PNG")
        buf.seek(0)

        # Generate public_id
        parsed_url = urlparse(image_url)
        original_name = os.path.basename(parsed_url.path)
        name_wo_ext = os.path.splitext(original_name)[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = uuid.uuid4().hex[:6]
        public_id = f"{name_wo_ext}_edit_{timestamp}_{random_suffix}"

        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(buf, public_id=public_id, resource_type="image")

        # Progress complete
        callback(100.0)
        task_status[task_id]["image_url"] = upload_result["secure_url"]

        return upload_result["secure_url"]

    except Exception as e:
        task_status[task_id]["progress"] = -1.0
        task_status[task_id]["image_url"] = None
        raise RuntimeError(f"Failed to generate/upload image: {e}")

# ================== API endpoints ==================
@app.post("/edit-image")
async def edit_image(request: ImageRequest):
    task_id = str(uuid.uuid4())
    task_status[task_id] = {"progress": 0.0, "image_url": None}

    future = executor.submit(generate_and_upload, task_id, request.image_url, request.prompt)
    try:
        cloud_url = future.result()
        return {"task_id": task_id, "image_url": cloud_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/progress/{task_id}")
async def check_progress(task_id: str):
    if task_id not in task_status:
        return JSONResponse(status_code=404, content={"error": "task_id not found"})
    
    status = task_status[task_id]
    response = {"task_id": task_id, "progress": status["progress"]}
    
    if status["progress"] == 100.0 and status["image_url"]:
        response["image_url"] = status["image_url"]
    
    return response

# ================== Run server ==================
if __name__ == "__main__":
    uvicorn.run("runqwen:app", host="0.0.0.0", port=7860, reload=False)
