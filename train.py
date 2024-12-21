def load_specific_checkpoint(checkpoint_path, model, optimizer=None, map_location=None):
    """Load a specific checkpoint file with validation"""
    logger = logging.getLogger(__name__)
    
    # Verify file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    try:
        if map_location is None:
            map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Verify checkpoint structure
        required_keys = ['epoch', 'batch', 'model_state_dict']
        if not all(k in checkpoint for k in required_keys):
            raise ValueError(f"Checkpoint missing required keys: {[k for k in required_keys if k not in checkpoint]}")
        
        # Load model state
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Successfully loaded model state")
        except Exception as e:
            raise RuntimeError(f"Failed to load model state: {str(e)}")
            
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Successfully loaded optimizer state")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {str(e)}")
                
        return checkpoint['epoch'], checkpoint['batch'], checkpoint.get('loss', 0)
        
    except Exception as e:
        logger.error(f"Error loading checkpoint {checkpoint_path}: {str(e)}")
        raise

def main():
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Specific checkpoint path
    checkpoint_path = './model_checkpoints/checkpoint_epoch_1_batch_500.pt'
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
        logger.info("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        model.gradient_checkpointing_enable()
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        # Load specific checkpoint
        try:
            start_epoch, start_batch, last_loss = load_specific_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                map_location='cuda'
            )
            start_batch += 1  # Start from next batch
            logger.info(f"Successfully loaded checkpoint. Resuming from epoch {start_epoch+1}, batch {start_batch}")
            logger.info(f"Last recorded loss: {last_loss}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            logger.info("Starting training from beginning...")
            start_epoch = 0
            start_batch = 0
        
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
        logger.info("Starting/Resuming training...")
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