from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, pipeline

app = Flask(__name__)
CORS(app)

# KoGPT 모델 로드
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token='</s>',
    eos_token='</s>',
    unk_token='<unk>',
    pad_token='<pad>',
    mask_token='<mask>'
)

text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 키워드 기반 링크 응답 딕셔너리
LINK_KNOWLEDGE_BASE = {
    "자연어처리": {
        "info": "자연어처리(NLP)는 인간의 언어를 컴퓨터가 이해하고 해석할 수 있도록 돕는 기술입니다.",
        "link": "https://wikidocs.net/216"
    },
    "딥러닝": {
        "info": "딥러닝은 인공신경망을 기반으로 하는 머신러닝 기법으로, 이미지 인식, 음성 처리, 자연어 이해 등에 폭넓게 사용됩니다.",
        "link": "https://tensorflow.blog/%EC%9E%90%EC%97%B0%EC%96%B4%EC%B2%98%EB%A6%AC/"
    },
    "BERT": {
        "info": "BERT는 트랜스포머 기반의 사전학습 모델로 문맥을 양방향으로 이해할 수 있습니다.",
        "link": "https://huggingface.co/bert-base-multilingual-cased"
    }
}

@app.route("/gpt", methods=["POST"])
def chat_with_korean_model():
    data = request.get_json()
    user_input = data.get("prompt", "")

    try:
        # 특정 키워드가 포함되면 사전 정의된 응답
        for keyword, content in LINK_KNOWLEDGE_BASE.items():
            if keyword in user_input:
                answer = f"{content['info']}\n\n🔗 관련 링크: {content['link']}"
                break
        else:
            # 일반 텍스트 생성
            result = text_generator(
                user_input,
                max_length=100,
                do_sample=True,
                temperature=0.7,
                top_k=1000,
                top_p=9.95,
                num_return_sequences=1
            )
            answer = result[0]['generated_text']

        return jsonify({
            "choices": [
                {"message": {"role": "assistant", "content": answer}}
            ]
        })

    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
