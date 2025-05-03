import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg to avoid display issues
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import urllib.request
import tarfile
from pathlib import Path
import glob
import argparse
import re
import shutil
import sys

# Handle Colab vs. Kaggle environment differences
IN_COLAB = 'google.colab' in sys.modules
IN_KAGGLE = 'kaggle_secrets' in sys.modules if not IN_COLAB else False

if IN_COLAB:
    # Colab directories
    BASE_DIR = Path('/content')
    MODEL_DIR = BASE_DIR / 'models'
    LOG_DIR = BASE_DIR / 'logs'
    WORKING_DIR = BASE_DIR / 'working'
else:
    # Kaggle directories - updated to use /kaggle/working for all outputs
    BASE_DIR = Path('/kaggle')
    WORKING_DIR = BASE_DIR / 'working'
    MODEL_DIR = WORKING_DIR / 'models'
    LOG_DIR = WORKING_DIR / 'logs'

# Create directories
WORKING_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Setup argument parser for checkpoint selection
def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Train a denoising autoencoder with checkpoint management')
    parser.add_argument('--resume-from', type=str, default=None, 
                        help='Checkpoint file to resume training from')
    parser.add_argument('--keep-checkpoints', type=int, default=3, 
                        help='Number of recent checkpoints to keep')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Batch size for training')
    
    # Parse only known arguments (ignores additional args passed in by Colab/Jupyter)
    return parser.parse_known_args()[0]

# Check for GPU availability and configure
def setup_gpus():
    """Configure TensorFlow to use multiple GPUs if available"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("No GPUs found. Running on CPU.")
        return False
    
    print(f"Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  - {gpu.name}")
    
    # REMOVED mixed precision policy to avoid 'Cast' layer issues
    # This is the key change to make the model more portable
    
    # Multi-GPU strategy
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} devices")
        return strategy
    else:
        print("Using default strategy (single GPU)")
        return tf.distribute.get_strategy()

def build_denoising_autoencoder(strategy=None, input_shape=(256, 256, 3)):
    """
    Build a professional-level U-Net style denoising autoencoder for images
    with multi-GPU support if available
    """
    if strategy:
        with strategy.scope():
            # Input layer
            inputs = layers.Input(shape=input_shape)
            
            # Encoder
            # Block 1
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x1 = x  # Skip connection 1
            x = layers.MaxPooling2D((2, 2))(x)
            
            # Block 2
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x2 = x  # Skip connection 2
            x = layers.MaxPooling2D((2, 2))(x)
            
            # Block 3
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x3 = x  # Skip connection 3
            x = layers.MaxPooling2D((2, 2))(x)
            
            # Block 4 (Bottleneck)
            x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Decoder
            # Block 5
            x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
            x = layers.Concatenate()([x, x3])  # Skip connection from Block 3
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Block 6
            x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
            x = layers.Concatenate()([x, x2])  # Skip connection from Block 2
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Block 7
            x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
            x = layers.Concatenate()([x, x1])  # Skip connection from Block 1
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Output layer
            outputs = layers.Conv2D(input_shape[2], (1, 1), activation='sigmoid')(x)
            
            # Create and compile model with Adam optimizer
            # Using standard optimizer settings without mixed precision
            optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
            
            # Create model
            model = models.Model(inputs, outputs)
            model.compile(
                optimizer=optimizer, 
                loss='mse', 
                metrics=['mae']
            )
    else:
        # If no strategy provided, build with default scope
        inputs = layers.Input(shape=input_shape)
        
        # Encoder
        # Block 1
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x1 = x  # Skip connection 1
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Block 2
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x2 = x  # Skip connection 2
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Block 3
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x3 = x  # Skip connection 3
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Block 4 (Bottleneck)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Decoder
        # Block 5
        x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Concatenate()([x, x3])  # Skip connection from Block 3
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Block 6
        x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Concatenate()([x, x2])  # Skip connection from Block 2
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Block 7
        x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.Concatenate()([x, x1])  # Skip connection from Block 1
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer
        outputs = layers.Conv2D(input_shape[2], (1, 1), activation='sigmoid')(x)
        
        # Create model
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer='adam', 
            loss='mse', 
            metrics=['mae']
        )
    
    return model

def download_BSD300_dataset():
    """
    Download BSD300 dataset for training/testing the model
    """
    data_dir = BASE_DIR / 'data'
    data_dir.mkdir(exist_ok=True, parents=True)
    
    # Define BSD300 dataset URL
    bsd_url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz"
    bsd_file = data_dir / "BSDS300-images.tgz"
    
    # Check if we're in Kaggle and dataset already exists in Kaggle input
    if IN_KAGGLE:
        input_path = Path('/kaggle/input/bsd300')
        if input_path.exists():
            print(f"Found BSD300 dataset in Kaggle input directory: {input_path}")
            return input_path
    
    # Download the dataset if it doesn't exist
    if not bsd_file.exists():
        print(f"Downloading BSD300 dataset from {bsd_url}...")
        urllib.request.urlretrieve(bsd_url, bsd_file)
        
        # Extract the archive
        with tarfile.open(bsd_file, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        print("Dataset downloaded and extracted successfully!")
    else:
        print("BSD300 dataset already exists in working directory.")
    
    return data_dir / "BSDS300" / "images"

def create_and_save_model(strategy=None):
    """
    Create the model with multi-GPU support and save it
    """
    print("Building denoising autoencoder model...")
    model = build_denoising_autoencoder(strategy)
    
    # Show model architecture summary
    model.summary()
    
    # Save the model architecture - using JSON for better portability
    model_json = model.to_json()
    with open(MODEL_DIR / "denoising_autoencoder_architecture.json", "w") as json_file:
        json_file.write(model_json)
    print("Model architecture saved to denoising_autoencoder_architecture.json")
    
    # Save the initial model weights
    model.save_weights(MODEL_DIR / "denoising_autoencoder.00-0.0000.weights.h5")
    print("Initial model weights saved to denoising_autoencoder_weights.h5")
    
    # Save the complete model - using SavedModel format for better portability
    model.save(MODEL_DIR / "denoising_autoencoder_model.keras")  # SavedModel format by default with directory path

    print("Complete model saved to denoising_autoencoder_model in TF SavedModel format")
    
    # Also save in H5 format for backwards compatibility
    model.save(MODEL_DIR / "denoising_autoencoder_model.h5")  # H5 format based on .h5 extension
    print("Complete model also saved to denoising_autoencoder_model.h5")
    
    return model

def load_model_from_checkpoint(checkpoint_path, strategy=None):
    """
    Load model from a specific checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
        strategy: TensorFlow distribution strategy
    
    Returns:
        Loaded model with weights from checkpoint
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # First, create the model with the right architecture
    if strategy:
        with strategy.scope():
            model = build_denoising_autoencoder(None)  # Strategy is already handled in the scope
    else:
        model = build_denoising_autoencoder()
    
    # Load weights from checkpoint
    model.load_weights(checkpoint_path)
    print("Model weights loaded successfully")
    
    # Extract epoch number from checkpoint filename for resuming training
    epoch_match = re.search(r'\.(\d+)-', os.path.basename(checkpoint_path))
    initial_epoch = 0
    if epoch_match:
        initial_epoch = int(epoch_match.group(1))
        print(f"Will resume training from epoch {initial_epoch}")
    
    return model, initial_epoch

def manage_checkpoints(keep_count=3, started_checkpoint=None):
    """
    Manage checkpoints to save disk space
    
    Args:
        keep_count: Number of recent checkpoints to keep
        started_checkpoint: Path to the checkpoint we started with (always keep this one)
    """
    # Get all checkpoint files in the model directory
    checkpoint_pattern = str(MODEL_DIR / "denoising_autoencoder.*.weights.h5")
    checkpoints = glob.glob(checkpoint_pattern)
    
    # If no checkpoints or too few to manage, just return
    if len(checkpoints) <= keep_count:
        return
    
    print(f"Managing checkpoints. Keeping {keep_count} most recent checkpoints...")
    
    # Extract epoch and loss info from filenames to sort
    checkpoint_info = []
    for cp in checkpoints:
        # Skip the checkpoint we started from
        if started_checkpoint and os.path.basename(cp) == os.path.basename(started_checkpoint):
            continue
            
        # Extract epoch number for sorting
        epoch_match = re.search(r'\.(\d+)-', os.path.basename(cp))
        if epoch_match:
            epoch = int(epoch_match.group(1))
            checkpoint_info.append((cp, epoch))
    
    # Sort by epoch (descending)
    checkpoint_info.sort(key=lambda x: x[1], reverse=True)
    
    # Keep the top 'keep_count' checkpoints, delete the rest
    checkpoints_to_delete = checkpoint_info[keep_count:]
    for cp, _ in checkpoints_to_delete:
        print(f"Removing old checkpoint: {os.path.basename(cp)}")
        os.remove(cp)

def train_model(model, train_images, val_images, args, initial_epoch=0):
    """
    Train the denoising autoencoder model with optimized settings
    and checkpoint management
    
    Args:
        model: The model to train
        train_images: Training images
        val_images: Validation images
        args: Command line arguments
        initial_epoch: Epoch to resume training from
    
    Returns:
        Trained model and training history
    """
    # Set epochs and batch size
    epochs = args.epochs
    batch_size = args.batch_size
    
    # Calculate appropriate batch size based on GPU RAM
    # For multi-GPU setup, we can increase batch size
    # but we need to make it divisible by number of GPUs
    gpus = len(tf.config.list_physical_devices('GPU'))
    if gpus > 1:
        # Make batch size divisible by number of GPUs
        # and scale it up for multiple GPUs
        batch_size = max(batch_size, 32 * gpus)
        # Ensure divisibility
        batch_size = batch_size - (batch_size % gpus) if batch_size % gpus != 0 else batch_size
        print(f"Using batch size of {batch_size} for {gpus} GPUs")
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=MODEL_DIR / "denoising_autoencoder.{epoch:02d}-{val_loss:.4f}.weights.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss'
    )
    
    # TensorBoard callback for visualizing training
    tensorboard_callback = TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    # Custom callback for checkpoint management
    class CheckpointManagerCallback(tf.keras.callbacks.Callback):
        def __init__(self, keep_count, started_checkpoint):
            super().__init__()
            self.keep_count = keep_count
            self.started_checkpoint = started_checkpoint
            
        def on_epoch_end(self, epoch, logs=None):
            # Manage checkpoints every 5 epochs to avoid too frequent disk operations
            if epoch % 5 == 0:
                manage_checkpoints(self.keep_count, self.started_checkpoint)
    
    # Create the checkpoint manager callback
    checkpoint_manager = CheckpointManagerCallback(
        args.keep_checkpoints,
        args.resume_from
    )
    
    # Create noisy versions of the training and validation images
    def add_noise(images, noise_factor=0.3):
        noisy_images = images + noise_factor * np.random.normal(
            loc=0.0, scale=1.0, size=images.shape
        )
        return np.clip(noisy_images, 0., 1.)
    
    train_noisy = add_noise(train_images)
    val_noisy = add_noise(val_images)
    
    # Train the model
    history = model.fit(
        train_noisy, train_images,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_noisy, val_images),
        callbacks=[checkpoint_callback, early_stopping, tensorboard_callback, checkpoint_manager],
        initial_epoch=initial_epoch
    )
    
    # Save the final trained model in both SavedModel and H5 formats for maximum compatibility
    # SavedModel format (recommended for TensorFlow 2.x)
    model.save(MODEL_DIR / "denoising_autoencoder_final.keras")  # SavedModel format

    print("Final model saved to denoising_autoencoder_final in TF SavedModel format")
    
    # Also save in H5 format for backward compatibility
    # Use 'tf' backend for saving to avoid mixed precision issues
    model.save(MODEL_DIR / "denoising_autoencoder_final.h5")  # H5 format
    print("Final model also saved to denoising_autoencoder_final.h5")
    
    # Save the training history for later analysis
    np.save(MODEL_DIR / 'training_history.npy', history.history)
    
    # Plot and save training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'training_curves.png', format='png')
    
    return model, history

def test_model(model):
    """
    Create a simple test to verify the model works
    """
    print("\nTesting model with a simple example...")
    # Create a sample clean image (all ones for simplicity)
    clean_img = np.ones((1, 256, 256, 3), dtype=np.float32) * 0.5
    
    # Add some noise
    noise_factor = 0.3
    noisy_img = clean_img + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=clean_img.shape).astype(np.float32)
    noisy_img = np.clip(noisy_img, 0., 1.)
    
    # Use the model to denoise
    denoised_img = model.predict(noisy_img)
    
    # Save the test images - ensure all arrays are float32 compatible with matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert arrays to float32 and clip to [0,1] range
    clean_display = np.clip(clean_img[0].astype(np.float32), 0, 1)
    noisy_display = np.clip(noisy_img[0].astype(np.float32), 0, 1) 
    denoised_display = np.clip(denoised_img[0].astype(np.float32), 0, 1)
    
    axes[0].imshow(clean_display)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(noisy_display)
    axes[1].set_title('Noisy')
    axes[1].axis('off')
    
    axes[2].imshow(denoised_display)
    axes[2].set_title('Denoised')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Try alternative method to save figure that's more reliable with float arrays
    plt.savefig(str(MODEL_DIR / 'test_results.png'), format='png', dpi=100)
    print("Test results saved to test_results.png")

def prepare_data_from_bsd300(data_path, image_size=(256, 256), split_ratio=0.8):
    """
    Prepare training and validation data from BSD300 dataset
    
    Args:
        data_path: Path to the BSD300 images directory
        image_size: Size to which images will be resized
        split_ratio: Ratio for train/validation split
    
    Returns:
        train_images, val_images as numpy arrays
    """
    # Find all images in the train folder
    train_path = data_path / "train"
    image_files = glob.glob(str(train_path / "*.jpg"))
    
    if not image_files:
        print(f"No images found in {train_path}. Using 'test' folder instead.")
        train_path = data_path / "test"
        image_files = glob.glob(str(train_path / "*.jpg"))
    
    if not image_files:
        raise ValueError(f"No images found in {data_path}")
    
    print(f"Found {len(image_files)} images for training/validation")
    
    # Load and preprocess images - optimize for memory
    # Process in batches to avoid memory issues
    batch_size = 32
    images = []
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_files:
            img = load_img(img_path, target_size=image_size)
            img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
            batch_images.append(img_array)
        
        images.extend(batch_images)
        print(f"Processed {min(i+batch_size, len(image_files))}/{len(image_files)} images")
    
    # Convert to numpy array
    images = np.array(images, dtype=np.float32)  # Explicitly use float32
    
    # Split into training and validation sets
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    print(f"Training images: {train_images.shape}")
    print(f"Validation images: {val_images.shape}")
    
    # Save preprocessed data to avoid reprocessing
    np.save(BASE_DIR / 'working/train_images.npy', train_images)
    np.save(BASE_DIR / 'working/val_images.npy', val_images)
    print("Preprocessed data saved to working directory")
    
    return train_images, val_images

def check_for_existing_data():
    """Check if preprocessed data already exists in input folder"""
    # Make paths based on environment
    if IN_KAGGLE:
        train_path = Path('/kaggle/input/preprocessed-bsd300/train_images.npy')
        val_path = Path('/kaggle/input/preprocessed-bsd300/val_images.npy')
    else:
        # For Colab, look in working directory
        train_path = BASE_DIR / 'working/train_images.npy'
        val_path = BASE_DIR / 'working/val_images.npy'
    
    if train_path.exists() and val_path.exists():
        print("Found preprocessed data in input folder")
        train_images = np.load(train_path)
        val_images = np.load(val_path)
        print(f"Loaded training images: {train_images.shape}")
        print(f"Loaded validation images: {val_images.shape}")
        return train_images, val_images
    
    return None, None

def main():
    print("Starting Image Denoising Autoencoder setup...")
    print(f"Running in {'Colab' if IN_COLAB else 'Kaggle' if IN_KAGGLE else 'other'} environment")
    
    # Parse command line arguments
    args = setup_arg_parser()
    
    # Setup GPUs and get strategy for multi-GPU training
    strategy = setup_gpus()
    
    # Check if preprocessed data already exists (useful for continuing work)
    train_images, val_images = check_for_existing_data()
    
    if train_images is None:
        # Download dataset if needed
        data_path = download_BSD300_dataset()
        print(f"Dataset available at: {data_path}")
        
        # Prepare data
        train_images, val_images = prepare_data_from_bsd300(data_path)
    
    # Initial epoch to start from
    initial_epoch = 0
    
    # Check if we're resuming from a checkpoint
    if args.resume_from:
        checkpoint_path = args.resume_from
        if not os.path.isabs(checkpoint_path):
            # If relative path, assume it's relative to MODEL_DIR
            checkpoint_path = str(MODEL_DIR / checkpoint_path)
        
        if os.path.exists(checkpoint_path):
            model, initial_epoch = load_model_from_checkpoint(checkpoint_path, strategy)
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Starting from scratch.")
            model = create_and_save_model(strategy)
    else:
        # Create and save model (with multi-GPU support)
        model = create_and_save_model(strategy)
    
    # Train model with checkpoint management
    model, history = train_model(model, train_images, val_images, args, initial_epoch)
    
    # Run a simple test and save results
    test_model(model)
    
    # Final checkpoint management
    manage_checkpoints(args.keep_checkpoints, args.resume_from)
    
    print("\nSetup complete!")
    print(f"Model and related files saved to: {MODEL_DIR}")
    print(f"Logs saved to: {LOG_DIR}")
    print("\nNext steps:")
    print("1. Download the model files")
    print("2. Use the SavedModel format at 'denoising_autoencoder_final' for best compatibility")
    print("3. Alternatively, use the H5 model at 'denoising_autoencoder_final.h5'")
    print("4. When loading the model, use tf.keras.models.load_model('/path/to/denoising_autoencoder_final')")
    print("   or model = tf.keras.models.load_model('/path/to/denoising_autoencoder_final.h5')")
    print("5. For best portability, avoid using mixed precision when loading and using the model")
    print("6. View training results in TensorBoard with: tensorboard --logdir={}".format(LOG_DIR))

if __name__ == "__main__":
    # Set the TF_FORCE_GPU_ALLOW_GROWTH environment variable to avoid memory issues
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Set memory growth for GPUs to avoid memory allocation errors
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device}")
        except:
            print(f"Could not set memory growth for {device}")
    
    # Run the main function
    main()