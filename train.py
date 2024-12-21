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

# def main():
#     # Initialize logging
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
    
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Using device: {device}")
    
#     # Model and tokenizer initialization
#     model_name = "Cognitive-Lab/LLama3-Gaja-Hindi-8B-v0.1"
#     logger.info("Loading model and tokenizer...")
    
#     try:
#         # Initialize tokenizer first
#         tokenizer = AutoTokenizer.from_pretrained(
#             model_name, 
#             trust_remote_code=True,
#             padding_side="left"  # Important for casual language modeling
#         )
        
#         # Ensure pad token is set
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
        
#         # Load model with memory optimizations
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.bfloat16,
#             device_map="auto",
#             low_cpu_mem_usage=True
#         )
#         model.gradient_checkpointing_enable()
        
#         # Load training data
#         train_data_path = 'training_data.jsonl'
#         logger.info(f"Loading training data from {train_data_path}")
#         train_data = load_training_data(train_data_path)
#         logger.info(f"Loaded {len(train_data)} conversation pairs")
        
#         # Create dataset and dataloader with proper collation
#         dataset = CustomDataset(train_data, tokenizer)
#         train_dataloader = DataLoader(
#             dataset, 
#             batch_size=1,  # Small batch size due to model size
#             shuffle=True,
#             collate_fn=default_data_collator
#         )
        
#         # Training parameters
#         num_epochs = 3
#         learning_rate = 1e-5
#         save_dir = './model_checkpoints'
        
#         # Start training
#         logger.info("Starting training...")
#         train_model(model, train_dataloader, num_epochs, learning_rate, device, save_dir)
#         logger.info("Training completed successfully!")
        
#         # Save final model
#         final_save_path = './fine_tuned_model'
#         os.makedirs(final_save_path, exist_ok=True)
#         model.save_pretrained(final_save_path)
#         tokenizer.save_pretrained(final_save_path)
#         logger.info(f"Model saved to {final_save_path}")
        
#     except Exception as e:
#         logger.error(f"An error occurred during training: {str(e)}")
#         raise  # Re-raise the exception for debugging


def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory"""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*_batch_*.pt'))
    if not checkpoints:
        return None
        
    # Extract epoch and batch numbers from filenames
    checkpoint_info = []
    for checkpoint in checkpoints:
        base = os.path.basename(checkpoint)
        try:
            # Extract epoch and batch numbers from filename
            parts = base.replace('.pt', '').split('_')
            epoch = int(parts[2])
            batch = int(parts[4])
            checkpoint_info.append((epoch, batch, checkpoint))
        except:
            continue
            
    if not checkpoint_info:
        return None
        
    # Sort by epoch and batch number
    checkpoint_info.sort(key=lambda x: (x[0], x[1]))
    return checkpoint_info[-1]  # Return the latest checkpoint

def load_training_state(model, optimizer, checkpoint_path):
    """Load model and optimizer state from checkpoint"""
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['batch'], checkpoint['loss']

def train_model(model, train_dataloader, num_epochs, learning_rate, device, save_dir, start_epoch=0, start_batch=0):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Load previous training state if exists
    latest_checkpoint = get_latest_checkpoint(save_dir)
    if latest_checkpoint:
        epoch_num, batch_num, checkpoint_path = latest_checkpoint
        start_epoch, start_batch, last_loss = load_training_state(model, optimizer, checkpoint_path)
        logging.info(f"Resuming from epoch {start_epoch+1}, batch {start_batch+1}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        # Skip batches that were already processed in the current epoch
        if epoch == start_epoch:
            for _ in range(start_batch):
                next(iter(progress_bar))
        
        for batch_idx, batch in enumerate(progress_bar, start=start_batch if epoch == start_epoch else 0):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Clear any leftover gradients
            model.zero_grad()
            
            try:
                # Forward pass with proper attention mask
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
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'loss': avg_loss,
                    'progress': f"{((batch_idx + 1) / len(train_dataloader)) * 100:.2f}%"
                })
                
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
                
            except RuntimeError as e:
                logging.error(f"Error during training: {str(e)}")
                logging.info("Saving emergency checkpoint...")
                
                # Save emergency checkpoint
                emergency_path = os.path.join(save_dir, f'emergency_checkpoint_epoch_{epoch+1}_batch_{batch_idx+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss / (batch_idx + 1) if batch_idx > 0 else 0,
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
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Paths
    checkpoint_dir = './model_checkpoints'
    model_name = "Cognitive-Lab/LLama3-Gaja-Hindi-8B-v0.1"
    
    try:
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        model.gradient_checkpointing_enable()
        
        # Find latest checkpoint
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        start_epoch = 0
        start_batch = 0
        
        if latest_checkpoint:
            epoch_num, batch_num, checkpoint_path = latest_checkpoint
            logger.info(f"Found checkpoint at epoch {epoch_num+1}, batch {batch_num+1}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = epoch_num
            start_batch = batch_num + 1  # Start from next batch
        
        # Load training data and create dataloader (same as before)
        train_data_path = 'training_data.jsonl'
        logger.info(f"Loading training data from {train_data_path}")
        train_data = load_training_data(train_data_path)
        logger.info(f"Loaded {len(train_data)} conversation pairs")
        
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
        
        # Resume training
        logger.info("Resuming training...")
        train_model(
            model, 
            train_dataloader, 
            num_epochs, 
            learning_rate, 
            device, 
            checkpoint_dir,
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