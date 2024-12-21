from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading
import time

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load the model and tokenizer
MODEL_NAME = "Cognitive-Lab/LLama3-Gaja-Hindi-8B-v0.1"
device = torch.device("mps")  # Change to "cuda" if using NVIDIA GPU

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("Model loaded!")

# Define the inference function
def generate_response_stream(prompt, max_new_tokens=256, temperature=0.6, top_p=0.9):
    print(f"Prompt: {prompt}")
    start_time = time.time()  # Start timing
    # Tokenize the input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Use TextIteratorStreamer for streaming responses
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    # Run generation in a separate thread
    def generate():
        model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            streamer=streamer,
        )

    thread = threading.Thread(target=generate)
    thread.start()

    # Buffer tokens into sentences
    buffer = ""
    for token in streamer:
        buffer += token
        if token.endswith(".") or token.endswith("?") or token.endswith("!"):
            yield buffer.strip()  # Yield the complete sentence
            buffer = ""
    if buffer:  # Emit any remaining text
        yield buffer.strip()
    end_time = time.time()  # End timing
    print(f"Inference completed in {end_time - start_time:.2f} seconds.")

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
        start_time = time.time()
        response = " ".join(generate_response_stream(prompt))
        end_time = time.time()

        return jsonify({
            "response": response,
            "time_taken": f"{end_time - start_time:.2f} seconds"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# WebSocket route for real-time streaming
@socketio.on("generate")
def handle_generate(data):
    try:
        prompt = data.get("prompt", "")
        if not prompt:
            emit("error", {"error": "Prompt is required"})
            return

        start_time = time.time()  # Start timing

        # Stream response back to the client sentence-wise
        for sentence in generate_response_stream(prompt):
            print(f"Sentence emitted: {sentence}")
            emit("response", {"sentence": sentence})

        end_time = time.time()  # End timing
        emit("response_complete", {
            "message": "Response generation complete.",
            "time_taken": f"{end_time - start_time:.2f} seconds"
        })

    except Exception as e:
        emit("error", {"error": str(e)})

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "up"})

# Run the app
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8080)
