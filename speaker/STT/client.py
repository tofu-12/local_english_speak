import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class STTClient:
    def __init__(self, device: torch.device):
        # 変数の設定
        torch_dtype=torch.float32
        model_id = "openai/whisper-large-v3-turbo"

        # モデルの初期化
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        ).to(device)

        # プロセッサーの初期化
        processor = AutoProcessor.from_pretrained(model_id)

        # pipelineの初期化
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

    def run(self, audio_file_path: str) -> str:
        result = self.pipe(
            audio_file_path,
            generate_kwargs={"language": "en"}
        )
        return result["text"]


if __name__ == "__main__":
    # デバイスの取得
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # 実行
    client = STTClient(device)
    text = client.run("test.mp3")
    print(text)
