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
from pathlib import Path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate audio list file from a directory of wav files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing wav files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the generated list file",
    )
    return parser.parse_args()


def extract_text_from_filename(filename):
    """从文件名中提取文本内容"""
    # 假设文件名格式为 raw.wav_start_end.wav
    # 如果文件名不包含文本内容，则返回文件名
    return os.path.splitext(filename)[0]


def generate_audio_list(input_dir, output_dir):
    """生成音频列表文件"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 获取目录名作为列表文件名
    dir_name = input_path.name
    list_file = output_path / f"{dir_name}.list"
    
    # 确保输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Processing directory: {input_dir}")
    logging.info(f"Output file: {list_file}")
    
    # 获取所有wav文件
    wav_files = sorted([f for f in input_path.glob("*.wav")])
    
    if not wav_files:
        logging.warning(f"No wav files found in {input_dir}")
        return
    
    # 写入列表文件
    with open(list_file, "w", encoding="utf-8") as f:
        for wav_file in wav_files:
            # 获取绝对路径
            abs_path = str(wav_file.absolute())
            # 获取文件名（不包含扩展名）作为文本内容
            text = extract_text_from_filename(wav_file.name)
            # 写入格式化的行
            line = f"{abs_path}|{dir_name}|ZH|{text}\n"
            f.write(line)
    
    logging.info(f"Successfully processed {len(wav_files)} files")
    logging.info(f"List file generated at: {list_file}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    args = parse_args()
    generate_audio_list(args.input_dir, args.output_dir) 


    # python -m cli.batch_inference --input_file "D:\downloads\refs.txt" --save_dir example/results --prompt_speech_path "D:\aisound\GPT-SoVITS\configs\refsounds\宝卿\不能跟自己和解，不能好好爱自己，就很难给别人温暖 与爱。.wav" --prompt_text "不能跟自己和解，不能好好爱自己，就很难给别人温暖与爱。"


    # python -m cli.generate_audio_list --input_dir "D:\aisound\Spark-TTS\example\宝卿" --output_dir "example"