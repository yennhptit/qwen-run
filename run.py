import io
import uuid
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
from diffusers.utils import load_image
from PIL import Image
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()
CLOUD_NAME = os.getenv("CLOUD_NAME")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=API_KEY,
    api_secret=API_SECRET,
)

# FastAPI setup
app = FastAPI(title="Qwen Image Edit CPU Only")

class ImageRequest(BaseModel):
    image_url: str
    prompt: str

MAX_WORKERS = 2
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
progress_dict: Dict[str, float] = {}

# CPU device
device = "cpu"

# Model setup
model_id = "Qwen/Qwen-Image-Edit"
torch_dtype = torch.bfloat16

quantization_config = DiffusersBitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["transformer_blocks.0.img_mod"]
)
transformer = QwenImageTransformer2DModel.from_pretrained(
    model_id, subfolder="transformer", quantization_config=quantization_config, torch_dtype=torch_dtype
).to(device)

quantization_config = TransformersBitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, subfolder="text_encoder", quantization_config=quantization_config, torch_dtype=torch_dtype
).to(device)

pipe = QwenImageEditPipeline.from_pretrained(
    model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
)
pipe.load_lora_weights(
    "flymy-ai/qwen-image-edit-inscene-lora",
    weight_name="flymy_qwen_image_edit_inscene_lora.safetensors"
)
pipe.enable_model_cpu_offload()

generator = torch.Generator(device=device).manual_seed(42)

def generate_and_upload(task_id: str, image_url: str, prompt: str) -> str:
    try:
        def callback(step, timestep, latents):
            total_steps = 32
            percent = (step + 1) / total_steps * 100
            progress_dict[task_id] = percent

        image = load_image(image_url)
        edited_image = pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=32,
            generator=generator,
            callback=callback,
            callback_steps=1
        ).images[0]

        buf = io.BytesIO()
        edited_image.save(buf, format="PNG")
        buf.seek(0)

        public_id = f"qwen_{uuid.uuid4().hex}"
        upload_result = cloudinary.uploader.upload(buf, public_id=public_id, resource_type="image")
        progress_dict[task_id] = 100.0
        return upload_result["secure_url"]
    except Exception as e:
        progress_dict[task_id] = -1.0
        raise RuntimeError(f"Failed to generate/upload image: {e}")

@app.post("/edit-image")
async def edit_image(request: ImageRequest):
    task_id = str(uuid.uuid4())
    progress_dict[task_id] = 0.0
    future = executor.submit(generate_and_upload, task_id, request.image_url, request.prompt)
    try:
        cloud_url = future.result()
        return {"task_id": task_id, "image_url": cloud_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/progress/{task_id}")
async def check_progress(task_id: str):
    if task_id not in progress_dict:
        return JSONResponse(status_code=404, content={"error": "task_id not found"})
    percent = progress_dict[task_id]
    return {"task_id": task_id, "progress": percent}
