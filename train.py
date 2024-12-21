import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
import logging
import os
from tqdm import tqdm
import json
import glob
import gc
import traceback

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
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
                "labels": encoded[0].clone()
            }
        except Exception as e:
            logging.error(f"Error processing item {idx}: {str(e)}")
            # Return a dummy batch in case of error
            return {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
                "labels": torch.zeros(self.max_length, dtype=torch.long)
            }

def load_training_data(file_path):
    conversations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    conversation = json.loads(line.strip())
                    if validate_conversation(conversation, line_num):
                        conversations.append(conversation)
                except Exception as e:
                    logging.warning(f"Line {line_num}: Error: {str(e)}")
                    continue
    
    if not conversations:
        raise ValueError("No valid conversations loaded from file")
    
    logging.info(f"Successfully loaded {len(conversations)} conversations")
    return conversations

def validate_conversation(conversation, line_num):
    """Validate conversation format"""
    if len(conversation) != 2:
        logging.warning(f"Line {line_num}: Expected 2 messages, got {len(conversation)}")
        return False
        
    if not all(isinstance(msg, dict) for msg in conversation):
        logging.warning(f"Line {line_num}: Invalid message format")
        return False
        
    if not all({"role", "content"} <= msg.keys() for msg in conversation):
        logging.warning(f"Line {line_num}: Missing required fields")
        return False
        
    if conversation[0]["role"] != "user" or conversation[1]["role"] != "assistant":
        logging.warning(f"Line {line_num}: Invalid roles")
        return False
        
    return True

def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    """Load checkpoint with proper error handling"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint['epoch'], checkpoint['batch'], checkpoint.get('loss', 0)
    except Exception as e:
        logging.error(f"Error loading checkpoint: {str(e)}\n{traceback.format_exc()}")
        raise

def save_checkpoint(model, optimizer, epoch, batch_idx, loss, save_path):
    """Save checkpoint with error handling"""
    try:
        torch.save({
            'epoch': epoch,
            'batch': batch_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, save_path)
    except Exception as e:
        logging.error(f"Error saving checkpoint: {str(e)}")
        # Try saving just the model state if full checkpoint fails
        try:
            torch.save({
                'epoch': epoch,
                'batch': batch_idx,
                'model_state_dict': model.state_dict(),
                'loss': loss,
            }, save_path)
        except Exception as e2:
            logging.error(f"Critical error saving checkpoint: {str(e2)}")
            raise

def train_model(model, train_dataloader, num_epochs, learning_rate, device, save_dir, 
                start_epoch=0, start_batch=0):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        for epoch in range(start_epoch, num_epochs):
            total_loss = 0
            batch_count = 0
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            # Skip processed batches if resuming
            if epoch == start_epoch and start_batch > 0:
                for _ in range(start_batch):
                    next(iter(progress_bar))
            
            for batch_idx, batch in enumerate(progress_bar, start=start_batch if epoch == start_epoch else 0):
                try:
                    # Move batch to device
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    # Clear gradients
                    model.zero_grad()
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update weights
                    optimizer.step()
                    
                    # Update metrics
                    total_loss += loss.item()
                    batch_count += 1
                    
                    # Update progress bar
                    avg_loss = total_loss / batch_count
                    progress = (batch_idx + 1) / len(train_dataloader) * 100
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'progress': f'{progress:.2f}%'
                    })
                    
                    # Save checkpoint every 100 batches
                    if (batch_idx + 1) % 100 == 0:
                        checkpoint_path = os.path.join(
                            save_dir, 
                            f'checkpoint_epoch_{epoch+1}_batch_{batch_idx+1}.pt'
                        )
                        save_checkpoint(
                            model, optimizer, epoch, batch_idx, avg_loss, checkpoint_path
                        )
                    
                    # Clear memory
                    del outputs, loss, input_ids, attention_mask, labels
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}: {str(e)}\n{traceback.format_exc()}")
                    # Save emergency checkpoint
                    emergency_path = os.path.join(
                        save_dir,
                        f'emergency_checkpoint_epoch_{epoch+1}_batch_{batch_idx+1}.pt'
                    )
                    save_checkpoint(
                        model, optimizer, epoch, batch_idx, 
                        total_loss / batch_count if batch_count > 0 else 0,
                        emergency_path
                    )
                    raise
            
            # Save epoch checkpoint
            epoch_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pt')
            save_checkpoint(
                model, optimizer, epoch, len(train_dataloader)-1,
                total_loss / batch_count if batch_count > 0 else 0,
                epoch_path
            )
            
            logging.info(f'Epoch {epoch+1} completed. Average loss: {total_loss / batch_count if batch_count > 0 else 0}')
            
    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Saving checkpoint...")
        interrupt_path = os.path.join(save_dir, 'interrupt_checkpoint.pt')
        save_checkpoint(model, optimizer, epoch, batch_idx, avg_loss, interrupt_path)
        raise
    
    except Exception as e:
        logging.error(f"Training error: {str(e)}\n{traceback.format_exc()}")
        raise

def main():
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Configuration
        model_name = "Cognitive-Lab/LLama3-Gaja-Hindi-8B-v0.1"
        checkpoint_path = './model_checkpoints/checkpoint_epoch_1_batch_500.pt'
        train_data_path = 'training_data.jsonl'
        save_dir = './model_checkpoints'
        num_epochs = 3
        learning_rate = 1e-5
        
        # Initialize tokenizer
        logger.info("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        model.gradient_checkpointing_enable()
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Load checkpoint if exists
        start_epoch = 0
        start_batch = 0
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            start_epoch, start_batch, _ = load_checkpoint(
                checkpoint_path, model, optimizer, device
            )
            start_batch += 1  # Start from next batch
            logger.info(f"Resuming from epoch {start_epoch+1}, batch {start_batch}")
        
        # Load training data
        logger.info("Loading training data...")
        train_data = load_training_data(train_data_path)
        
        # Create dataset and dataloader
        dataset = CustomDataset(train_data, tokenizer)
        train_dataloader = DataLoader(
            dataset, 
            batch_size=1,
            shuffle=False,
            collate_fn=default_data_collator
        )
        
        # Start training
        logger.info("Starting training...")
        train_model(
            model,
            train_dataloader,
            num_epochs,
            learning_rate,
            device,
            save_dir,
            start_epoch,
            start_batch
        )
        
        # Save final model
        logger.info("Saving final model...")
        final_save_path = './fine_tuned_model'
        os.makedirs(final_save_path, exist_ok=True)
        model.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        logger.info(f"Model saved to {final_save_path}")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()