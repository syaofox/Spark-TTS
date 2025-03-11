import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from cli.SparkTTS import SparkTTS


def load_characters_config(config_path):
    """加载角色配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device=0):
    """初始化模型"""
    logging.info(f"Loading model from: {model_dir}")
    device = torch.device(f"cuda:{device}")
    model = SparkTTS(Path(model_dir), device)
    return model


def run_tts(text, model, prompt_text, prompt_speech, save_dir="example/results"):
    """执行TTS推理并保存生成的音频"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")

    with torch.no_grad():
        # 如果 prompt_text 为空，则只使用 prompt_speech 进行推理
        if not prompt_text:
            wav = model.inference(text, prompt_speech)
        else:
            wav = model.inference(text, prompt_speech, prompt_text)

        sf.write(save_path, wav, samplerate=16000)

    return save_path


def build_character_ui(model_dir, config_path, device=0):
    # 初始化模型
    model = initialize_model(model_dir, device=device)

    # 加载角色配置
    characters_config = load_characters_config(config_path)

    def synthesize_text(character, emotion, input_text):
        """单次合成回调函数"""
        if not character or not emotion or not input_text:
            return None

        try:
            char_config = characters_config[character][emotion]
            prompt_text = char_config["text"]
            prompt_speech = char_config["audio"]

            return run_tts(
                input_text, model, prompt_text=prompt_text, prompt_speech=prompt_speech
            )
        except Exception as e:
            logging.error(f"合成失败: {str(e)}")
            return None

    def parse_line(line: str) -> Tuple[str | None, str | None, str]:
        """解析包含角色和情绪的文本行
        格式: (角色|情绪)文本内容
        返回: (角色, 情绪, 文本内容)
        """
        pattern = r"^\(([^|]+)\|([^)]+)\)(.+)$"
        match = re.match(pattern, line.strip())
        if match:
            char, emotion, text = match.groups()
            return char.strip(), emotion.strip(), text.strip()
        return None, None, line.strip()

    def merge_audio_files(
        audio_files: List[str], gap_duration: float = 0.3
    ) -> str | None:
        """合并多个音频文件，在空行处添加间隔"""
        if not audio_files:
            return None

        waves = []
        sample_rate = 16000  # 采样率
        gap = np.zeros(int(gap_duration * sample_rate))  # 生成间隔静音

        for i, file in enumerate(audio_files):
            if file == "gap":
                waves.append(gap)
            else:
                wave, _ = sf.read(file)
                waves.append(wave)

        # 合并所有音频
        merged_wave = np.concatenate(waves)

        # 保存合并后的音频
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = os.path.join("example/results", f"merged_{timestamp}.wav")
        sf.write(output_path, merged_wave, sample_rate)

        return output_path

    def batch_synthesize(input_texts):
        """批量合成回调函数"""
        if not input_texts.strip():
            return None

        try:
            # 分行处理文本
            lines = input_texts.split("\n")
            audio_files = []

            for line in lines:
                line = line.strip()
                if not line:  # 空行添加间隔标记
                    audio_files.append("gap")
                    continue

                # 解析行内容
                char, emotion, text = parse_line(line)
                if not char or not emotion:
                    logging.warning(f"跳过格式不正确的行: {line}")
                    continue

                if (
                    char not in characters_config
                    or emotion not in characters_config[char]
                ):
                    logging.warning(f"无效的角色或情绪: {char}|{emotion}")
                    continue

                # 合成音频
                audio_path = synthesize_text(char, emotion, text)
                if audio_path:
                    audio_files.append(audio_path)

            # 合并所有音频文件
            return merge_audio_files(audio_files)

        except Exception as e:
            logging.error(f"批量合成失败: {str(e)}")
            return None

    with gr.Blocks() as demo:
        gr.HTML('<h1 style="text-align: center;">角色语音合成系统</h1>')

        # 使用选项卡分隔单行和批量合成
        with gr.Tabs():
            with gr.Tab("单行合成"):
                with gr.Row():
                    with gr.Column():
                        # 角色选择
                        default_character = (
                            list(characters_config.keys())[0]
                            if characters_config
                            else None
                        )
                        character = gr.Dropdown(
                            choices=list(characters_config.keys()),
                            label="选择角色",
                            value=default_character,
                        )

                        # 情绪选择
                        default_emotions = (
                            list(characters_config[default_character].keys())
                            if default_character
                            else []
                        )
                        emotion = gr.Dropdown(
                            choices=default_emotions,
                            label="选择情绪",
                            value=default_emotions[0] if default_emotions else None,
                        )

                        def update_emotions(char_name):
                            """更新情绪选项的回调函数"""
                            if not char_name or char_name not in characters_config:
                                return gr.Dropdown(choices=[], value=None)

                            emotions = list(characters_config[char_name].keys())
                            return gr.Dropdown(
                                choices=emotions,
                                value=emotions[0] if emotions else None,
                            )

                with gr.Row():
                    # 单行合成
                    single_text = gr.Textbox(
                        label="文本内容", placeholder="输入要合成的文本", value=""
                    )

                with gr.Row():
                    single_btn = gr.Button("合成", variant="primary")
                    single_audio = gr.Audio(label="合成结果")

            with gr.Tab("批量合成"):
                gr.Markdown("""### 使用说明
                每行文本格式为 (角色|情绪)文本内容
                - 例如：(京京|开心)今天天气真好！
                - 空行将添加 0.3 秒间隔
                """)

                with gr.Row():
                    batch_text = gr.TextArea(
                        label="多行文本",
                        placeholder="(角色|情绪)文本内容\n(角色|情绪)文本内容",
                        lines=8,
                        value="",
                    )

                with gr.Row():
                    batch_btn = gr.Button("批量合成", variant="primary")
                    batch_audio = gr.Audio(label="合成结果")

        # 当角色改变时更新情绪列表
        character.change(
            fn=update_emotions,
            inputs=character,
            outputs=emotion,
        )

        # 绑定单行合成按钮事件
        single_btn.click(
            fn=synthesize_text,
            inputs=[character, emotion, single_text],
            outputs=single_audio,
        )

        # 绑定批量合成按钮事件
        batch_btn.click(
            fn=batch_synthesize,
            inputs=[batch_text],
            outputs=batch_audio,
        )

    return demo


def parse_arguments():
    parser = argparse.ArgumentParser(description="Character TTS Gradio server")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="模型目录路径",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="characters/config.json",
        help="角色配置文件路径",
    )
    parser.add_argument("--device", type=int, default=0, help="GPU设备ID")
    parser.add_argument(
        "--server_name", type=str, default="0.0.0.0", help="服务器主机名/IP"
    )
    parser.add_argument("--server_port", type=int, default=7860, help="服务器端口")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    demo = build_character_ui(
        model_dir=args.model_dir, config_path=args.config_path, device=args.device
    )

    # 使用最简单的启动配置
    demo.launch()
