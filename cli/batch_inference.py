# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import logging
import os
import re
from datetime import datetime

import soundfile as sf
import torch

from cli.SparkTTS import SparkTTS


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch TTS inference with voice cloning."
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="example/results",
        help="Directory to save generated audio files",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device number")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input text file with one sentence per line",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        required=True,
        help="Transcript of reference audio for voice cloning",
    )
    parser.add_argument(
        "--prompt_speech_path",
        type=str,
        required=True,
        help="Path to the reference audio file for voice cloning",
    )
    parser.add_argument(
        "--pitch", choices=["very_low", "low", "moderate", "high", "very_high"]
    )
    parser.add_argument(
        "--speed", choices=["very_low", "low", "moderate", "high", "very_high"]
    )
    return parser.parse_args()


def sanitize_filename(text):
    """将文本转换为合法的文件名"""
    # 移除或替换不合法的字符
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    # 限制文件名长度
    return text[:150] if len(text) > 150 else text


def run_batch_tts(args):
    """执行批量TTS推理并保存生成的音频"""
    logging.info(f"Using model from: {args.model_dir}")
    logging.info(f"Reading texts from: {args.input_file}")
    logging.info(f"Using reference audio: {args.prompt_speech_path}")
    logging.info(f"Reference transcript: {args.prompt_text}")
    logging.info(f"Saving audio to: {args.save_dir}")

    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)

    # 确保参考音频文件存在
    if not os.path.exists(args.prompt_speech_path):
        logging.error(f"Reference audio file not found: {args.prompt_speech_path}")
        return

    # 设置设备
    device = torch.device(f"cuda:{args.device}")

    # 初始化模型
    model = SparkTTS(args.model_dir, device)

    # 读取输入文件
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logging.error(f"Error reading input file: {e}")
        return

    logging.info(f"Found {len(texts)} lines to process")

    # 处理每一行文本
    for i, text in enumerate(texts, 1):
        try:
            # 生成文件名
            filename = sanitize_filename(text)
            save_path = os.path.join(args.save_dir, f"{filename}.wav")

            logging.info(f"Processing line {i}/{len(texts)}: {text[:50]}...")

            # 执行推理并保存音频
            with torch.no_grad():
                wav = model.inference(
                    text,
                    args.prompt_speech_path,  # 使用指定的参考音频
                    prompt_text=args.prompt_text,  # 使用参考音频的文本
                    pitch=args.pitch,
                    speed=args.speed,
                )
                sf.write(save_path, wav, samplerate=16000)

            logging.info(f"Audio saved at: {save_path}")

        except Exception as e:
            logging.error(f"Error processing line {i}: {e}")
            continue


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_args()
    run_batch_tts(args)
