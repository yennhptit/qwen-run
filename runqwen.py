import io
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from PIL import Image
from urllib.parse import urlparse
import torch
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import uvicorn
import time
import os

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
app = FastAPI(title="Qwen Image Edit Background")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== ThreadPool & task tracking ==================
MAX_WORKERS = 2
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
task_status: Dict[str, Dict] = {}  # task_id -> {"progress": float, "image_url": str}

# ================== Device & Model setup ==================
device = "cuda" if torch.cuda.is_available() else "cpu"
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

# ================== Background task ==================
def generate_and_upload(task_id: str, image: Image.Image, prompt: str):
    try:
        # Fake progress mượt theo 32 steps
        num_steps = 32
        for step in range(num_steps):
            task_status[task_id]["progress"] = (step / num_steps) * 80
            time.sleep(0.1)  # giả lập

        # Chạy pipeline thật
        edited_image = pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=num_steps,
            generator=generator
        ).images[0]

        # Save to buffer
        buf = io.BytesIO()
        edited_image.save(buf, format="PNG")
        buf.seek(0)

        # Generate public_id
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = uuid.uuid4().hex[:6]
        public_id = f"edit_{timestamp}_{random_suffix}"

        # Upload Cloudinary
        upload_result = cloudinary.uploader.upload(buf, public_id=public_id, resource_type="image")

        # Update progress complete
        task_status[task_id]["progress"] = 100.0
        task_status[task_id]["image_url"] = upload_result["secure_url"]

    except Exception as e:
        task_status[task_id]["progress"] = -1.0
        task_status[task_id]["image_url"] = None
        print(f"Task {task_id} failed: {e}")

# ================== API endpoints ==================
@app.post("/edit-image-file")
async def edit_image_file(file: UploadFile = File(...), prompt: str = "Add a cat"):
    task_id = str(uuid.uuid4())
    task_status[task_id] = {"progress": 0.0, "image_url": None}

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    executor.submit(generate_and_upload, task_id, image, prompt)

    # Trả về ngay task_id
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

# ================== Run server ==================
if __name__ == "__main__":
    uvicorn.run("runqwen:app", host="0.0.0.0", port=7860, reload=False)
