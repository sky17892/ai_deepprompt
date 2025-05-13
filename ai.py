from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, pipeline
from dotenv import load_dotenv
import os

# 환경 변수 로드 12345
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

app = Flask(__name__)
CORS(app, origins=["https://html-starter-wheat-iota.vercel.app"])
#CORS(app)

# 모델 및 토크나이저 로드 (환경변수로 토큰 전달)
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2", use_auth_token=HF_TOKEN)
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token="</s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
    use_auth_token=HF_TOKEN
)

text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

LINK_KNOWLEDGE_BASE = {
    "자연어처리": {
        "info": "자연어처리(NLP)는 인간의 언어를 컴퓨터가 이해하고 해석할 수 있도록 돕는 기술입니다.",
        "link": "https://wikidocs.net/216"
    },
    "딥러닝": {
        "info": "딥러닝은 인공신경망 기반의 머신러닝 기법으로 다양한 AI 분야에 활용됩니다.",
        "link": "https://tensorflow.blog/%EC%9E%90%EC%97%B0%EC%96%B4%EC%B2%98%EB%A6%AC/"
    },
    "BERT": {
        "info": "BERT는 트랜스포머 구조의 사전학습 언어모델로 문장의 문맥을 양방향으로 이해합니다.",
        "link": "https://huggingface.co/bert-base-multilingual-cased"
    }
}

@app.route("/gpt", methods=["POST"])
def generate_response():
    data = request.get_json()
    user_input = data.get("prompt", "")

    try:
        for keyword, content in LINK_KNOWLEDGE_BASE.items():
            if keyword in user_input:
                answer = f"{content['info']}\n\n🔗 관련 링크: {content['link']}"
                break
        else:
            result = text_generator(
                user_input,
                max_length=100,
                do_sample=True,
                temperature=0.7,
                top_k=1000,
                top_p=9.95,
                num_return_sequences=1
            )
            answer = result[0]["generated_text"]

        return jsonify({
            "choices": [
                {"message": {"role": "assistant", "content": answer}}
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
