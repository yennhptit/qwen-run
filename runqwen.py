from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from PIL import Image
import io
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
import time
from datetime import datetime
import cloudinary
import cloudinary.uploader
import torch
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
import os
from dotenv import load_dotenv
import uvicorn

# Load Cloudinary
load_dotenv()
cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("API_KEY"),
    api_secret=os.getenv("API_SECRET"),
)

# FastAPI
app = FastAPI(title="Qwen Image Edit via URL")

MAX_WORKERS = 2
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
task_status: Dict[str, Dict] = {}

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Qwen/Qwen-Image-Edit"
torch_dtype = torch.bfloat16

# Load models (same as before)
transformer_quant = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["transformer_blocks.0.img_mod"]
)
transformer = QwenImageTransformer2DModel.from_pretrained(
    model_id, subfolder="transformer", quantization_config=transformer_quant, torch_dtype=torch_dtype
).to(device)

text_encoder_quant = TransformersBitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, subfolder="text_encoder", quantization_config=text_encoder_quant, torch_dtype=torch_dtype
).to(device)

pipe = QwenImageEditPipeline.from_pretrained(
    model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
)
pipe.load_lora_weights("flymy_qwen_image_edit_inscene_lora.safetensors")
pipe.enable_model_cpu_offload()
generator = torch.Generator(device=device).manual_seed(42)

# Background task
def generate_and_upload(task_id: str, image: Image.Image, prompt: str):
    try:
        num_steps = 32
        for step in range(num_steps):
            task_status[task_id]["progress"] = (step / num_steps) * 80
            time.sleep(0.1)  # fake progress

        edited_image = pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=num_steps,
            generator=generator
        ).images[0]

        buf = io.BytesIO()
        edited_image.save(buf, format="PNG")
        buf.seek(0)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = uuid.uuid4().hex[:6]
        public_id = f"edit_{timestamp}_{random_suffix}"

        upload_result = cloudinary.uploader.upload(buf, public_id=public_id, resource_type="image")

        task_status[task_id]["progress"] = 100.0
        task_status[task_id]["image_url"] = upload_result["secure_url"]

    except Exception as e:
        task_status[task_id]["progress"] = -1.0
        task_status[task_id]["image_url"] = None
        print(f"Task {task_id} failed: {e}")

# ================== API endpoint nhận URL ==================
class ImageURLRequest(BaseModel):
    image_url: str
    prompt: str = "Add a cat"

@app.post("/edit-image-url")
async def edit_image_url(request: ImageURLRequest):
    task_id = str(uuid.uuid4())
    task_status[task_id] = {"progress": 0.0, "image_url": None}

    try:
        resp = requests.get(request.image_url, timeout=10)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")

    executor.submit(generate_and_upload, task_id, image, request.prompt)
    return {"task_id": task_id}

@app.get("/progress/{task_id}")
async def check_progress(task_id: str):
    if task_id not in task_status:
        return JSONResponse(status_code=404, content={"error": "task_id not found"})
    status = task_status[task_id]
    response = {"task_id": task_id, "progress": status["progress"]}
    if status["progress"] == 100.0 and status["image_url"]:
        response["image_url"] = status["image_url"]
    return response
if __name__ == "__main__": 
    uvicorn.run("runqwen:app", host="0.0.0.0", port=7860, reload=False)