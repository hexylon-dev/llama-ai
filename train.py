import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
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
        conversation = self.data[idx]
        
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in conversation
        ]
        
        # Apply chat template with proper padding
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Create attention mask
        attention_mask = (encoded != self.tokenizer.pad_token_id).long()
        
        return {
            "input_ids": encoded[0],
            "attention_mask": attention_mask[0],
            "labels": encoded[0].clone()  # Use same sequence for labels
        }

def load_training_data(file_path):
    """Load training data where each line is a JSON array containing a conversation pair"""
    conversations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    conversation = json.loads(line.strip())
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
    
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Clear any leftover gradients
            model.zero_grad()
            
            # Forward pass with proper attention mask
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
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
                
            # Clear memory
            del outputs, loss
            torch.cuda.empty_cache()
        
        # Save model after each epoch
        model_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), model_path)
        
        print(f'Epoch {epoch+1} completed. Average loss: {total_loss / len(train_dataloader)}')

def main():
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Model and tokenizer initialization
    model_name = "Cognitive-Lab/LLama3-Gaja-Hindi-8B-v0.1"
    logger.info("Loading model and tokenizer...")
    
    try:
        # Initialize tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            padding_side="left"  # Important for casual language modeling
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with memory optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        model.gradient_checkpointing_enable()
        
        # Load training data
        train_data_path = 'training_data.jsonl'
        logger.info(f"Loading training data from {train_data_path}")
        train_data = load_training_data(train_data_path)
        logger.info(f"Loaded {len(train_data)} conversation pairs")
        
        # Create dataset and dataloader with proper collation
        dataset = CustomDataset(train_data, tokenizer)
        train_dataloader = DataLoader(
            dataset, 
            batch_size=1,  # Small batch size due to model size
            shuffle=True,
            collate_fn=default_data_collator
        )
        
        # Training parameters
        num_epochs = 3
        learning_rate = 1e-5
        save_dir = './model_checkpoints'
        
        # Start training
        logger.info("Starting training...")
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
        raise  # Re-raise the exception for debugging

if __name__ == "__main__":
    main()