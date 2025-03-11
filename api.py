import os
import torch
import soundfile as sf
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Spark-TTS API", description="API for Spark Text-to-Speech synthesis")

# 全局变量
model = None
save_dir = "example/results"

def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device=0):
    """Load the model once at the beginning."""
    logging.info(f"Loading model from: {model_dir}")
    device = torch.device(f"cuda:{device}")
    model = SparkTTS(model_dir, device)
    return model

def run_tts(
    text,
    model,
    prompt_text=None,
    prompt_speech=None,
    gender=None,
    pitch=None,
    speed=None,
    save_dir="example/results",
):
    """Perform TTS inference and save the generated audio."""
    if prompt_text is not None:
        prompt_text = None if len(prompt_text) <= 1 else prompt_text
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")
    logging.info(f"prompt: {prompt_text}")
    logging.info(f"text: {text}")
    # Perform inference and save the output audio
    with torch.no_grad():
        wav = model.inference(
            text,
            prompt_speech,
            prompt_text,
            gender,
            pitch,
            speed,
        )
        sf.write(save_path, wav, samplerate=16000)
    logging.info(f"Audio saved at: {save_path}")
    return save_path

@app.on_event("startup")
async def startup_event():
    global model
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Spark TTS FastAPI server.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="ID of the GPU device to use (e.g., 0 for cuda:0)."
    )
    args, _ = parser.parse_known_args()
    
    # 初始化模型
    model = initialize_model(model_dir=args.model_dir, device=args.device)
    
    # 确保输出目录存在
    os.makedirs(save_dir, exist_ok=True)

@app.post("/voice_clone")
async def voice_clone(
    text: str = Form(...),
    prompt_text: Optional[str] = Form(None),
    prompt_audio: UploadFile = File(...)
):
    """
    Clone a voice based on reference audio.
    
    Args:
        text: The text to synthesize
        prompt_text: Text of the prompt speech (optional)
        prompt_audio: Reference audio file for voice cloning
    
    Returns:
        Audio file of synthesized speech
    """
    # 保存上传的音频文件
    prompt_audio_path = f"temp_{prompt_audio.filename}"
    with open(prompt_audio_path, "wb") as f:
        f.write(await prompt_audio.read())
    
    try:
        # 处理提示文本
        prompt_text_clean = None if prompt_text is None or len(prompt_text) < 2 else prompt_text
        
        # 执行TTS
        audio_output_path = run_tts(
            text,
            model,
            prompt_text=prompt_text_clean,
            prompt_speech=prompt_audio_path
        )
        
        # 返回生成的音频文件
        return FileResponse(
            audio_output_path, 
            media_type="audio/wav", 
            filename=os.path.basename(audio_output_path)
        )
    finally:
        # 清理临时文件
        if os.path.exists(prompt_audio_path):
            os.remove(prompt_audio_path)

@app.post("/voice_creation")
async def voice_creation(
    text: str = Form(...),
    gender: str = Form(...),
    pitch: int = Form(3),
    speed: int = Form(3)
):
    """
    Create a synthetic voice with adjustable parameters.
    
    Args:
        text: The text to synthesize
        gender: 'male' or 'female'
        pitch: Value from 1-5
        speed: Value from 1-5
    
    Returns:
        Audio file of synthesized speech
    """
    # 验证参数
    if gender not in ["male", "female"]:
        raise HTTPException(status_code=400, detail="Gender must be 'male' or 'female'")
    
    if not 1 <= pitch <= 5 or not 1 <= speed <= 5:
        raise HTTPException(status_code=400, detail="Pitch and speed must be between 1 and 5")
    
    # 映射参数
    pitch_val = LEVELS_MAP_UI[int(pitch)]
    speed_val = LEVELS_MAP_UI[int(speed)]
    
    # 执行TTS
    audio_output_path = run_tts(
        text,
        model,
        gender=gender,
        pitch=pitch_val,
        speed=speed_val
    )
    
    # 返回生成的音频文件
    return FileResponse(
        audio_output_path, 
        media_type="audio/wav", 
        filename=os.path.basename(audio_output_path)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)