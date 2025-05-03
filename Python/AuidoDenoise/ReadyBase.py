import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import matplotlib
matplotlib.use('Agg')  # Set backend to avoid display issues
import matplotlib.pyplot as plt
import urllib.request
import tarfile
import zipfile
from pathlib import Path
import glob
import argparse
import re
import shutil
import sys
import librosa
import soundfile as sf

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
    # Kaggle directories - updated to use /kaggle/working for outputs
    BASE_DIR = Path('/kaggle')
    WORKING_DIR = BASE_DIR / 'working'
    MODEL_DIR = WORKING_DIR / 'models'
    LOG_DIR = WORKING_DIR / 'logs'

# Create directories
WORKING_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Configuration
SAMPLE_RATE = 16000  # Standard speech sampling rate
FRAME_LENGTH = 2048  # Length of audio frames
HOP_LENGTH = 512     # Hop length between frames

def setup_arg_parser():
    """Setup argument parser for training configuration"""
    parser = argparse.ArgumentParser(description='Train an audio denoising autoencoder')
    parser.add_argument('--resume-from', type=str, default=None, 
                        help='Checkpoint file to resume training from')
    parser.add_argument('--keep-checkpoints', type=int, default=3, 
                        help='Number of recent checkpoints to keep')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Batch size for training')
    
    return parser.parse_known_args()[0]

def setup_gpus():
    """Configure TensorFlow to use multiple GPUs if available"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("No GPUs found. Running on CPU.")
        return False
    
    print(f"Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  - {gpu.name}")
    
    # Multi-GPU strategy
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} devices")
        return strategy
    else:
        print("Using default strategy (single GPU)")
        return tf.distribute.get_strategy()

def build_audio_denoising_autoencoder(strategy=None, input_shape=(128, 128, 1)):
    """
    Build a professional-level 2D CNN-based audio denoising autoencoder
    with multi-GPU support if available
    """
    if strategy:
        with strategy.scope():
            inputs = layers.Input(shape=input_shape)
            
            # Encoder
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            
            # Bottleneck
            x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Decoder
            x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
            x = layers.BatchNormalization()(x)
            
            # Output layer
            outputs = layers.Conv2D(1, (1, 1), activation='linear')(x)
            
            model = models.Model(inputs, outputs)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='mse',
                metrics=['mae']
            )
    else:
        # Standard model creation without strategy
        inputs = layers.Input(shape=input_shape)
        
        # Encoder (similar structure as with strategy)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Bottleneck
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Decoder
        x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer
        outputs = layers.Conv2D(1, (1, 1), activation='linear')(x)
        
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
    
    return model

def download_vctk_dataset():
    """
    Download VCTK dataset for audio denoising training
    
    Uses the compact version of VCTK dataset
    """
    data_dir = BASE_DIR / 'data'
    data_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if we're in Kaggle and dataset already exists in Kaggle input
    if IN_KAGGLE:
        input_path = Path('/kaggle/input/vctk-corpus')
        if input_path.exists():
            print(f"Found VCTK dataset in Kaggle input directory: {input_path}")
            return input_path
    
    # VCTK dataset URL (compact version)
    vctk_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus-0.92.zip"
    vctk_file = data_dir / "VCTK-Corpus-0.92.zip"
    
    # Download the dataset if it doesn't exist
    if not vctk_file.exists():
        print(f"Downloading VCTK dataset from {vctk_url}...")
        urllib.request.urlretrieve(vctk_url, vctk_file)
        
        # Extract the archive
        with zipfile.ZipFile(vctk_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Dataset downloaded and extracted successfully!")
    else:
        print("VCTK dataset already exists in working directory.")
    
    return data_dir / "VCTK-Corpus-0.92"

def prepare_audio_data(data_path, target_sr=SAMPLE_RATE):
    """
    Prepare audio data by extracting spectrograms, normalizing,
    and splitting into train/validation sets
    """
    def extract_spectrograms(audio_files, noise_factor=0.05):
        spectrograms = []
        clean_spectrograms = []
        
        for file_path in audio_files:
            try:
                # Load audio file
                audio, sr = librosa.load(file_path, sr=target_sr)
                
                # Add noise
                noisy_audio = audio + noise_factor * np.random.normal(0, 1, len(audio))
                
                # Compute spectrograms
                clean_spec = np.abs(librosa.stft(audio, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
                noisy_spec = np.abs(librosa.stft(noisy_audio, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
                
                # Normalization and scaling
                clean_spec = librosa.amplitude_to_db(clean_spec, ref=np.max)
                noisy_spec = librosa.amplitude_to_db(noisy_spec, ref=np.max)
                
                # Normalize to [0, 1]
                clean_spec = (clean_spec - clean_spec.min()) / (clean_spec.max() - clean_spec.min())
                noisy_spec = (noisy_spec - noisy_spec.min()) / (noisy_spec.max() - noisy_spec.min())
                
                spectrograms.append(noisy_spec.T[:128, :128, np.newaxis])
                clean_spectrograms.append(clean_spec.T[:128, :128, np.newaxis])
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return np.array(spectrograms), np.array(clean_spectrograms)
    
    # Find all wav files
    wav_files = []
    for root, _, files in os.walk(data_path):
        wav_files.extend([os.path.join(root, f) for f in files if f.endswith('.wav')])
    
    print(f"Total audio files found: {len(wav_files)}")
    
    # Limit dataset size to prevent memory issues
    wav_files = wav_files[:500]  # Adjust based on Kaggle memory constraints
    
    # Randomly shuffle files
    np.random.shuffle(wav_files)
    
    # Split into training and validation sets
    split_ratio = 0.8
    split_idx = int(len(wav_files) * split_ratio)
    
    train_files = wav_files[:split_idx]
    val_files = wav_files[split_idx:]
    
    # Extract spectrograms
    train_noisy, train_clean = extract_spectrograms(train_files)
    val_noisy, val_clean = extract_spectrograms(val_files)
    
    print(f"Training spectrograms shape: {train_noisy.shape}")
    print(f"Validation spectrograms shape: {val_noisy.shape}")
    
    # Optional: save preprocessed data
    np.save(WORKING_DIR / 'train_noisy_specs.npy', train_noisy)
    np.save(WORKING_DIR / 'train_clean_specs.npy', train_clean)
    np.save(WORKING_DIR / 'val_noisy_specs.npy', val_noisy)
    np.save(WORKING_DIR / 'val_clean_specs.npy', val_clean)
    
    return train_noisy, train_clean, val_noisy, val_clean

def check_for_existing_data():
    """Check if preprocessed data already exists"""
    data_paths = [
        WORKING_DIR / 'train_noisy_specs.npy',
        WORKING_DIR / 'train_clean_specs.npy',
        WORKING_DIR / 'val_noisy_specs.npy',
        WORKING_DIR / 'val_clean_specs.npy'
    ]
    
    if all(path.exists() for path in data_paths):
        print("Found preprocessed spectrograms")
        train_noisy = np.load(data_paths[0])
        train_clean = np.load(data_paths[1])
        val_noisy = np.load(data_paths[2])
        val_clean = np.load(data_paths[3])
        return train_noisy, train_clean, val_noisy, val_clean
    
    return None, None, None, None

def train_model(model, train_noisy, train_clean, val_noisy, val_clean, args):
    """
    Train the audio denoising model with checkpoint management
    """
    epochs = args.epochs
    batch_size = args.batch_size
    
    # Adjust batch size for multi-GPU
    gpus = len(tf.config.list_physical_devices('GPU'))
    if gpus > 1:
        batch_size = max(batch_size, 32 * gpus)
        batch_size = batch_size - (batch_size % gpus) if batch_size % gpus != 0 else batch_size
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        filepath=MODEL_DIR / "audio_denoiser.{epoch:02d}-{val_loss:.4f}.weights.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss'
    )
    
    tensorboard_callback = TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    # Checkpoint management callback
    checkpoint_manager_callback = CheckpointManagerCallback(
        args.keep_checkpoints,
        args.resume_from
    )
    
    history = model.fit(
        train_noisy, train_clean,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_noisy, val_clean),
        callbacks=[
            checkpoint_callback, 
            early_stopping, 
            tensorboard_callback,
            checkpoint_manager_callback
        ]
    )
    
    # Save final model
    model.save(MODEL_DIR / "audio_denoiser_final.keras")
    model.save(MODEL_DIR / "audio_denoiser_final.h5")
    
    # Save training history
    np.save(MODEL_DIR / 'training_history.npy', history.history)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'training_curves.png')
    
    return model, history

def test_model(model):
    """
    Test the audio denoising model with a sample spectrogram
    """
    # Create a simple test spectrogram
    test_spec = np.random.random((1, 128, 128, 1))
    
    # Add noise
    noisy_spec = test_spec + 0.1 * np.random.normal(0, 1, test_spec.shape)
    
    # Predict denoised spectrogram
    denoised_spec = model.predict(noisy_spec)
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(test_spec[0, :, :, 0], cmap='viridis')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_spec[0, :, :, 0], cmap='viridis')
    plt.title('Noisy')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(denoised_spec[0, :, :, 0], cmap='viridis')
    plt.title('Denoised')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'test_denoising_results.png')
    print("Test results saved to test_denoising_results.png")

def check_for_checkpoint(initial_checkpoint=None, strategy=None):
    """
    Check and load checkpoint if available
    """
    if initial_checkpoint:
        try:
            model, initial_epoch = load_model_from_checkpoint(initial_checkpoint, strategy)
            return model, initial_epoch
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
    
    return create_and_save_model(strategy), 0

def create_and_save_model(strategy=None):
    """Create and save initial model"""
    model = build_audio_denoising_autoencoder(strategy)
    model.summary()
    
    # Save model architecture
    model_json = model.to_json()
    with open(MODEL_DIR / "audio_denoiser_architecture.json", "w") as json_file:
        json_file.write(model_json)
    
    model.save_weights(MODEL_DIR / "audio_denoiser_initial.weights.h5")
    model.save(MODEL_DIR / "audio_denoiser_initial.keras")
    model.save(MODEL_DIR / "audio_denoiser_initial.h5")
    
    return model

def load_model_from_checkpoint(checkpoint_path, strategy=None):
    """Load model from checkpoint"""
    model = build_audio_denoising_autoencoder(strategy)
    model.load_weights(checkpoint_path)
    
    # Extract initial epoch
    epoch_match = re.search(r'\.(\d+)-', os.path.basename(checkpoint_path))
    initial_epoch = int(epoch_match.group(1)) if epoch_match else 0
    
    return model, initial_epoch

def manage_checkpoints(keep_count=3, started_checkpoint=None):
    """Manage model checkpoints"""
    checkpoint_pattern = str(MODEL_DIR / "audio_denoiser.*.weights.h5")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if len(checkpoints) <= keep_count:
        return
    
    checkpoint_info = []
    for cp in checkpoints:
        if started_checkpoint and os.path.basename(cp) == os.path.basename(started_checkpoint):
            continue
        
        epoch_match = re.search(r'\.(\d+)-', os.path.basename(cp))
        if epoch_match:
            epoch = int(epoch_match.group(1))
            checkpoint_info.append((cp, epoch))
    
    checkpoint_info.sort(key=lambda x: x[1], reverse=True)
    
    # Delete older checkpoints
    for cp, _ in checkpoint_info[keep_count:]:
        os.remove(cp)

class CheckpointManagerCallback(tf.keras.callbacks.Callback):
    """Callback to manage checkpoints during training"""
    def __init__(self, keep_count, started_checkpoint):
        super().__init__()
        self.keep_count = keep_count
        self.started_checkpoint = started_checkpoint
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            manage_checkpoints(self.keep_count, self.started_checkpoint)

def main():
    print("Starting Audio Denoising Autoencoder setup...")
    
    # Parse arguments
    args = setup_arg_parser()
    
    # Setup GPU strategy
    strategy = setup_gpus()
    
    # Check for existing preprocessed data
    train_noisy, train_clean, val_noisy, val_clean = check_for_existing_data()
    
    if train_noisy is None:
        # Download dataset
        data_path = download_vctk_dataset()
        print(f"Dataset available at: {data_path}")
        
        # Prepare data
        train_noisy, train_clean, val_noisy, val_clean = prepare_audio_data(data_path)
    
    # Check for checkpoint and create/load model
    model, initial_epoch = check_for_checkpoint(args.resume_from, strategy)
    
    # Train model
    model, history = train_model(
        model, 
        train_noisy, train_clean, 
        val_noisy, val_clean, 
        args
    )
    
    # Test model
    test_model(model)
    
    # Final checkpoint management
    manage_checkpoints(args.keep_checkpoints, args.resume_from)
    
    print("\nSetup complete!")
    print(f"Model files saved to: {MODEL_DIR}")
    print(f"Logs saved to: {LOG_DIR}")

if __name__ == "__main__":
    # Set memory growth for GPUs
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except Exception as e:
            print(f"Could not set memory growth for {device}: {e}")
    
    main()