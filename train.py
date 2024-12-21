import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
from tqdm import tqdm
import json

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        conversation = self.data[idx]  # This is now a list of user and assistant messages
        
        # Convert the conversation pair into messages format
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in conversation
        ]
        
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        if encoded.shape[1] > self.max_length:
            encoded = encoded[:, :self.max_length]
            
        return encoded[0]

def load_training_data(file_path):
    """Load training data where each line is a JSON array containing a conversation pair"""
    conversations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():  # Skip empty lines
                try:
                    # Parse the JSON array from the line
                    conversation = json.loads(line.strip())
                    
                    # Verify the conversation structure
                    if len(conversation) != 2:
                        logging.warning(f"Line {line_num}: Expected 2 messages, got {len(conversation)}. Skipping.")
                        continue
                        
                    if conversation[0]["role"] != "user" or conversation[1]["role"] != "assistant":
                        logging.warning(f"Line {line_num}: Invalid roles. Expected user-assistant pair. Skipping.")
                        continue
                        
                    conversations.append(conversation)
                    
                except json.JSONDecodeError as e:
                    logging.warning(f"Line {line_num}: Invalid JSON: {str(e)}")
                    continue
                except Exception as e:
                    logging.warning(f"Line {line_num}: Error processing line: {str(e)}")
                    continue
                    
    logging.info(f"Successfully loaded {len(conversations)} conversations")
    return conversations

def train_model(model, train_dataloader, num_epochs, learning_rate, device, save_dir):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.to(device)
            
            # Forward pass
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            optimizer.zero_grad()
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
            
            # Save checkpoint every 100 batches
            if (batch_idx + 1) % 100 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}_batch_{batch_idx+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                
        # Save model after each epoch
        model_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), model_path)
        
        print(f'Epoch {epoch+1} completed. Average loss: {total_loss / len(train_dataloader)}')

def main():
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set device
    device = torch.device("cuda")
    logger.info(f"Using device: {device}")
    
    # Model and tokenizer initialization
    model_name = "Cognitive-Lab/LLama3-Gaja-Hindi-8B-v0.1"
    logger.info("Loading model and tokenizer...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True
    ).to(device)
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load training data from file
    train_data_path = 'training_data.jsonl'  # Your JSONL file with conversation pairs
    logger.info(f"Loading training data from {train_data_path}")
    train_data = load_training_data(train_data_path)
    logger.info(f"Loaded {len(train_data)} conversation pairs")
    
    # Create dataset and dataloader
    dataset = CustomDataset(train_data, tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Training parameters
    num_epochs = 3
    learning_rate = 1e-5
    save_dir = './model_checkpoints'
    
    # Start training
    logger.info("Starting training...")
    try:
        train_model(model, train_dataloader, num_epochs, learning_rate, device, save_dir)
        logger.info("Training completed successfully!")
        
        # Save final model
        final_save_path = './fine_tuned_model'
        os.makedirs(final_save_path, exist_ok=True)
        model.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        logger.info(f"Model saved to {final_save_path}")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    main()