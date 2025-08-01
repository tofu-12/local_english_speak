import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MAX_CONVERSATION_HISTORY_LENGTH = 10


class LLMClient:
    def __init__(self, device: torch.device):
        # モデルの初期化
        model_id = "Qwen/Qwen2.5-3B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto"
        ).to(device)

        # tokenizerの初期化
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # promptの変数
        self.system_prompt = {
            "role": "system", 
            "content": "You are English teacher in Japan. Let's practice English conversation together."
        }
        self.conversation = []
    
    def run(self, prompt: str) -> str:
        # 履歴の削除
        self._delete_old_conversation()

        # メッセージの作成
        user_prompt = {
            "role": "user", 
            "content": prompt
        }
        self.conversation.append(user_prompt)
        messages = [self.system_prompt] + self.conversation

        # 応答の生成
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 応答の保存
        assistant_prompt = {
            "role": "assistant", 
            "content": response
        }
        self.conversation.append(assistant_prompt)

        return response
    
    def _delete_old_conversation(self) -> None:
        """ 古い会話履歴を消すメソッド """
        if len(self.conversation) > MAX_CONVERSATION_HISTORY_LENGTH:
            self.conversation = self.conversation[-MAX_CONVERSATION_HISTORY_LENGTH:]


if __name__ == "__main__":
    # デバイスの取得
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # 実行
    client = LLMClient(device)
    text = client.run("Hello, I'm Tom. I like soccer.")
    print(text)
