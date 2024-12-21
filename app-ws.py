from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading
import time
import requests

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load the model and tokenizer
MODEL_NAME = "Cognitive-Lab/LLama3-Gaja-Hindi-8B-v0.1"
device = torch.device("cuda")  # Change to "cuda" if using NVIDIA GPU

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("Model loaded!")

# Function to call external API and process response
def query_external_api(prompt):
    try:
        response = requests.post(
            'http://localhost:3000/query',
            json={'query': prompt},
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get('success') and data.get('results'):
            # Extract texts only from results with score < 1.25
            context_texts = []
            for result in data['results']:
                if ('metadata' in result and 
                    'text' in result['metadata'] and 
                    'score' in result and 
                    result['score'] < 1.80):
                    context_texts.append(result['metadata']['text'])
            
            # If we have any relevant texts, combine them
            if context_texts:
                combined_context = " ".join(context_texts)
                return combined_context
            return None
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error calling external API: {str(e)}")
        return None

def create_enhanced_prompt(original_prompt, context):
    """Create a prompt that combines context with the original query"""
    if context:
        return f"""Context: {context}

User Query: {original_prompt}

Aur strictly hindi aur english mix me response generate karna

Response:"""
    return original_prompt

# Define the inference function
def generate_response_stream(prompt, max_new_tokens=256, temperature=0.6, top_p=0.9):
    print(f"Original prompt received: {prompt}")
    
    # Get context from external API
    context = query_external_api(prompt)
    
    # Create enhanced prompt with context
    enhanced_prompt = create_enhanced_prompt(prompt, context)
    print(f"Enhanced prompt with context: {enhanced_prompt}")
    
    # Tokenize the input
    input_ids = tokenizer(enhanced_prompt, return_tensors="pt").input_ids.to(device)

    # Use TextIteratorStreamer for streaming responses
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    # Run generation in a separate thread
    def generate():
        model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            streamer=streamer,
        )

    thread = threading.Thread(target=generate)
    thread.start()

    # Collect tokens into chunks of 3-4 words
    buffer = []
    for token in streamer:
        buffer.append(token)
        if len(buffer) >= 3:  # Emit when 3-4 words are in the buffer
            yield " ".join(buffer)
            buffer = []
    if buffer:  # Emit any remaining words
        yield " ".join(buffer)

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
        response = " ".join(generate_response_stream(prompt))
        return jsonify({"response": response})

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

        # Stream response back to the client
        for chunk in generate_response_stream(prompt):
            print(f"Chunk emitted: {chunk}")
            emit("response", {"chunk": chunk}, namespace="/", broadcast=True)
            socketio.sleep(0)  # Allow async tasks to process
    
        emit("response_complete", {"message": "Response generation complete."})

    except Exception as e:
        emit("error", {"error": str(e)})

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "up"})

# Run the app
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8081)