from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
MODEL_NAME = "Cognitive-Lab/LLama3-Gaja-Hindi-8B-v0.1"
device = torch.device("mps")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("Model loaded!")

# Define the inference function
def generate_response(prompt, max_new_tokens=256, temperature=0.6, top_p=0.9):
    # Tokenize the input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    streamer = TextStreamer(tokenizer)

    # Generate response
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        streamer=streamer
    )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Define API routes
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the JSON data from the request
        data = request.json
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Generate response
        response = generate_response(prompt)
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "up"})

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
