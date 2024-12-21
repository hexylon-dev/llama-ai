import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
import logging
import os
from tqdm import tqdm
import json

# [Previous CustomDataset and load_training_data functions remain the same]
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

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load checkpoint with error handling"""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['batch'], checkpoint.get('loss', 0)
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        raise

def train_model(model, train_dataloader, num_epochs, learning_rate, device, save_dir, start_epoch=0, start_batch=0):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Load from checkpoint if specified
    if start_epoch > 0 or start_batch > 0:
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{start_epoch+1}_batch_{start_batch}.pt')
        try:
            start_epoch, start_batch, last_loss = load_checkpoint(checkpoint_path, model, optimizer, device)
            logging.info(f"Resuming from epoch {start_epoch+1}, batch {start_batch+1}")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {str(e)}")
            logging.info("Starting training from beginning")
            start_epoch = 0
            start_batch = 0
    
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        # Skip processed batches if resuming
        if epoch == start_epoch and start_batch > 0:
            logging.info(f"Skipping to batch {start_batch+1}")
            for _ in range(start_batch):
                try:
                    next(iter(progress_bar))
                except StopIteration:
                    break
        
        for batch_idx, batch in enumerate(progress_bar, start=start_batch if epoch == start_epoch else 0):
            try:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Clear any leftover gradients
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
                
                # Update progress bar
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx - start_batch + 1 if batch_idx >= start_batch else 1)
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
                    torch.save({
                        'epoch': epoch,
                        'batch': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                    }, checkpoint_path)
                
                # Clear memory
                del outputs, loss, input_ids, attention_mask, labels
                torch.cuda.empty_cache()
                
            except Exception as e:
                logging.error(f"Error in batch {batch_idx}: {str(e)}")
                # Save emergency checkpoint
                emergency_path = os.path.join(
                    save_dir,
                    f'emergency_checkpoint_epoch_{epoch+1}_batch_{batch_idx+1}.pt'
                )
                torch.save({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss if 'avg_loss' in locals() else 0,
                }, emergency_path)
                raise
        
        # Reset start_batch after first epoch
        start_batch = 0
        
        # Save model after each epoch
        model_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), model_path)
        
        print(f'Epoch {epoch+1} completed. Average loss: {total_loss / len(train_dataloader)}')

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
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Configuration
    model_name = "Cognitive-Lab/LLama3-Gaja-Hindi-8B-v0.1"
    checkpoint_path = './model_checkpoints/checkpoint_epoch_1_batch_500.pt'
    
    try:
        # Initialize tokenizer
        logger.info("Loading tokenizer...")
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
        
        # Initialize optimizer (needed for loading checkpoint)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        # Load checkpoint
        start_epoch = 0
        start_batch = 0
        if os.path.exists(checkpoint_path):
            logger.info(f"Found checkpoint at {checkpoint_path}")
            start_epoch, start_batch, _ = load_checkpoint(checkpoint_path, model, optimizer, device)
            start_batch += 1  # Start from next batch
            logger.info(f"Resuming from epoch {start_epoch+1}, batch {start_batch}")
        
        # Load training data
        train_data_path = 'training_data.jsonl'
        logger.info(f"Loading training data from {train_data_path}")
        train_data = load_training_data(train_data_path)
        logger.info(f"Loaded {len(train_data)} conversation pairs")
        
        # Create dataset and dataloader
        dataset = CustomDataset(train_data, tokenizer)
        train_dataloader = DataLoader(
            dataset, 
            batch_size=1,
            shuffle=False,  # Important: don't shuffle when resuming
            collate_fn=default_data_collator
        )
        
        # Training parameters
        num_epochs = 3
        learning_rate = 1e-5
        save_dir = './model_checkpoints'
        
        # Resume training
        logger.info("Resuming training...")
        train_model(
            model, 
            train_dataloader, 
            num_epochs, 
            learning_rate, 
            device, 
            save_dir,
            start_epoch=start_epoch,
            start_batch=start_batch
        )
        
        logger.info("Training completed successfully!")
        
        # Save final model
        final_save_path = './fine_tuned_model'
        os.makedirs(final_save_path, exist_ok=True)
        model.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        logger.info(f"Model saved to {final_save_path}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()