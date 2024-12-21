def verify_checkpoint_file(checkpoint_path):
    """Verify if checkpoint file is valid"""
    try:
        # Try to load the checkpoint file
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'batch']
        return all(key in checkpoint for key in required_keys)
    except Exception as e:
        logging.error(f"Error verifying checkpoint {checkpoint_path}: {str(e)}")
        return False

def find_latest_valid_checkpoint(checkpoint_dir):
    """Find the latest valid checkpoint in directory"""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*_batch_*.pt'))
    valid_checkpoints = []
    
    for checkpoint_path in checkpoints:
        try:
            if verify_checkpoint_file(checkpoint_path):
                # Extract epoch and batch numbers
                parts = os.path.basename(checkpoint_path).replace('.pt', '').split('_')
                epoch = int(parts[2])
                batch = int(parts[4])
                valid_checkpoints.append((epoch, batch, checkpoint_path))
        except Exception as e:
            logging.warning(f"Skipping invalid checkpoint {checkpoint_path}: {str(e)}")
            continue
    
    if not valid_checkpoints:
        return None
        
    # Sort by epoch and batch
    valid_checkpoints.sort(key=lambda x: (x[0], x[1]))
    return valid_checkpoints[-1]

def safe_load_checkpoint(model, optimizer, device='cuda'):
    """Safely load checkpoint with fallback options"""
    checkpoint_dir = './model_checkpoints'
    specified_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_epoch_1_batch_500.pt')
    
    try:
        # First try loading the specified checkpoint
        if os.path.exists(specified_checkpoint):
            logging.info(f"Attempting to load specified checkpoint: {specified_checkpoint}")
            checkpoint = torch.load(specified_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch'], checkpoint['batch'], checkpoint.get('loss', 0)
            
    except Exception as e:
        logging.warning(f"Failed to load specified checkpoint: {str(e)}")
        
    # If specified checkpoint fails, try finding latest valid checkpoint
    logging.info("Searching for latest valid checkpoint...")
    latest_valid = find_latest_valid_checkpoint(checkpoint_dir)
    
    if latest_valid:
        epoch_num, batch_num, checkpoint_path = latest_valid
        try:
            logging.info(f"Loading latest valid checkpoint from epoch {epoch_num+1}, batch {batch_num+1}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return epoch_num, batch_num, checkpoint.get('loss', 0)
        except Exception as e:
            logging.error(f"Error loading latest valid checkpoint: {str(e)}")
    
    # If all checkpoints fail, start from beginning
    logging.warning("No valid checkpoints found. Starting from beginning.")
    return 0, 0, 0

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
        
        # Initialize model and tokenizer
        model_name = "Cognitive-Lab/LLama3-Gaja-Hindi-8B-v0.1"
        
        # Load tokenizer
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
        
        # Try to load checkpoint
        start_epoch, start_batch, last_loss = safe_load_checkpoint(model, optimizer, device)
        
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
        
        # Start/Resume training
        if start_batch > 0 or start_epoch > 0:
            logger.info(f"Resuming training from epoch {start_epoch+1}, batch {start_batch+1}")
        else:
            logger.info("Starting training from beginning")
            
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
        logger.error(f"Fatal error: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()