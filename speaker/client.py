import os
import sys
import warnings

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from speaker.LLM.client import LLMClient
from speaker.STT.client import STTClient
from speaker.TTS.client import TTSClient


# 警告を非表示にする
warnings.simplefilter('ignore')


class Speaker:
    def __init__(self):
        self.conversation = []
        self.device = self._get_device()


    def speak(self, input_audio_path: str) -> str:
        """
        実行メソッド

        Args:
            input_audio_path: inputファイルパス
        
        Returns:
            str: outputファイルパス
        """
        # audio -> text
        client = STTClient(self.device)
        text = client.run(input_audio_path)

        # text -> text
        client = LLMClient(self.device)
        client.set_conversation(self.conversation)
        response = client.run(text)
        self.conversation = client.get_conversation()

        # text -> audio
        client = TTSClient(self.device)
        output_audio_path = client.run(response)

        return output_audio_path
    

    def _get_device(self) -> torch.device:
        """ 実行デバイスを取得するメソッド """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        return device
    

if __name__ == "__main__":
    input_audio_path = "test_data/test.mp3"
    speaker = Speaker()
    output_audio_path = speaker.speak(input_audio_path)

    print(output_audio_path)
