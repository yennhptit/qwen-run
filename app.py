import io
import os
import uuid
from datetime import datetime
from time import sleep
from threading import Thread
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict
from urllib.parse import urlparse
from PIL import Image
import requests
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
app = FastAPI(title="Qwen Image Edit Fake CPU Only")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả domain, test nhanh
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image_url: str
    prompt: str

# ================== Task tracking ==================
task_status: Dict[str, Dict] = {}  # task_id -> {"progress": float, "image_url": str}
MAX_STEP = 32
TOTAL_SECONDS = 60  # 1 phút

# ================== Fake image processing ==================
def fake_generate(task_id: str, image_url: str):
    step_delay = TOTAL_SECONDS / MAX_STEP
    for step in range(MAX_STEP):
        sleep(step_delay)
        task_status[task_id]["progress"] = (step + 1) / MAX_STEP * 100
    
    # Sau khi xong, lấy ảnh mặc định
    fixed_image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTc9APxkj0xClmrU3PpMZglHQkx446nQPG6lA&s"
    
    # Download ảnh fixed
    response = requests.get(fixed_image_url)
    buf = io.BytesIO(response.content)
    buf.seek(0)

    # Lấy tên file gốc từ URL và thêm _edit + timestamp + random suffix
    parsed_url = urlparse(image_url)
    original_name = os.path.basename(parsed_url.path)
    name_wo_ext = os.path.splitext(original_name)[0]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = uuid.uuid4().hex[:6]
    public_id = f"{name_wo_ext}_edit_{timestamp}_{random_suffix}"

    # Upload lên Cloudinary
    upload_result = cloudinary.uploader.upload(buf, public_id=public_id, resource_type="image")
    
    task_status[task_id]["progress"] = 100.0
    task_status[task_id]["image_url"] = upload_result["secure_url"]

# ================== API endpoints ==================
@app.post("/edit-image")
async def edit_image(request: ImageRequest):
    task_id = str(uuid.uuid4())
    task_status[task_id] = {"progress": 0.0, "image_url": None}

    # Chạy fake generation trong thread riêng
    Thread(target=fake_generate, args=(task_id, request.image_url), daemon=True).start()
    
    return {"task_id": task_id, "image_url": None}

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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
