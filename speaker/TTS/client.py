import os
from datetime import datetime

from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import torch
from transformers import AutoTokenizer


OUTPUT_DIR_PATH = os.path.join("conversation_data", "assistant")


class TTSClient:
    def __init__(self, device: torch.device):
        # 変数の設定
        model_id = "parler-tts/parler-tts-mini-v1"

        # モデルの初期化
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)

        # tokenizerの初期化
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # descriptionの設定
        self.description = (
            "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. "
            "The recording is of very high quality, with the speaker's voice sounding clear and very close up."
        )
    
    def run(self, prompt: str) -> str:
        """
        実行メソッド

        Args:
            prompt: プロンプト
        
        Returns:
            str: outputファイルパス
        """
        # トークンidの取得
        input_ids = self.tokenizer(self.description, return_tensors="pt").input_ids.to(self.model.device)
        prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        # 音声の生成
        generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()

        # 音声の保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_name = f"{timestamp}.wav"
        output_file_path = os.path.join(OUTPUT_DIR_PATH, output_file_name)

        sf.write(output_file_path, audio_arr, self.model.config.sampling_rate)
        
        return output_file_path


if __name__ == "__main__":
    # デバイスの取得
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # 実行
    client = TTSClient(device)
    output_file_path = client.run("Hello, I'm Tom. Nice to meet you.")
    print(output_file_path)
