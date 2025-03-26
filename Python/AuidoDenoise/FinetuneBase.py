import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Concatenate, BatchNormalization, Activation, Add, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from pathlib import Path
import random
import librosa
import soundfile as sf
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# Handle Colab vs. Kaggle environment differences
IN_COLAB = 'google.colab' in sys.modules
IN_KAGGLE = 'kaggle_secrets' in sys.modules if not IN_COLAB else False

# Configuration for directories
if IN_COLAB:
    BASE_DIR = Path('/content')
    MODEL_DIR = BASE_DIR / 'models'
    LOG_DIR = BASE_DIR / 'logs'
    WORKING_DIR = BASE_DIR / 'working'
else:
    BASE_DIR = Path('/kaggle')
    WORKING_DIR = BASE_DIR / 'working'
    MODEL_DIR = WORKING_DIR / 'models'
    LOG_DIR = WORKING_DIR / 'logs'
    INPUT_DIR = BASE_DIR / 'input'

# Create necessary directories
for directory in [WORKING_DIR, MODEL_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# GPU Strategy
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
strategy = tf.distribute.MirroredStrategy() if len(tf.config.list_physical_devices('GPU')) > 0 else tf.distribute.get_strategy()

# Audio Processing Utilities
def load_audio_files(dataset_path, max_duration=10, sr=16000, limit=None):
    """
    Load audio files from a specified dataset path
    
    Args:
        dataset_path: Path to the dataset
        max_duration: Maximum duration of audio clips in seconds
        sr: Sampling rate
        limit: Limit number of files to process
    
    Returns:
        List of audio numpy arrays
    """
    audio_files = list(Path(dataset_path).rglob('*.wav'))
    
    if limit:
        audio_files = audio_files[:limit]
    
    clean_audio_clips = []
    
    for audio_path in tqdm(audio_files, desc="Loading Audio Files"):
        try:
            # Load audio file
            audio, sample_rate = librosa.load(audio_path, sr=sr, duration=max_duration)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            clean_audio_clips.append(audio)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    
    return clean_audio_clips

def apply_noise(audio, noise_types=None, noise_params=None):
    """
    Apply different types of noise to an audio signal
    
    Args:
        audio: Normalized numpy audio array
        noise_types: List of noise types
        noise_params: Parameters for noise generation
    
    Returns:
        Noisy audio numpy array
    """
    if noise_types is None:
        noise_types = random.choice([
            ['gaussian'], 
            ['white'], 
            ['pink'], 
            ['impulse']
        ])
    
    if noise_params is None:
        noise_params = {}
    
    # Make a copy of the audio
    noisy_audio = audio.copy()
    
    for noise_type in noise_types:
        if noise_type == 'gaussian':
            std = noise_params.get('gaussian_std', random.choice([0.01, 0.05, 0.1]))
            noise = np.random.normal(0, std, audio.shape)
            noisy_audio += noise
        
        elif noise_type == 'white':
            amplitude = noise_params.get('white_amplitude', random.choice([0.01, 0.05, 0.1]))
            noise = np.random.uniform(-amplitude, amplitude, audio.shape)
            noisy_audio += noise
        
        elif noise_type == 'pink':
            # Simulate pink noise (1/f noise)
            n = len(audio)
            f = np.fft.fftfreq(n)
            pink_noise = np.random.normal(0, 1, n) / np.sqrt(np.abs(f) + 1e-10)
            pink_noise = np.real(np.fft.ifft(pink_noise)) * 0.1
            noisy_audio += pink_noise
        
        elif noise_type == 'impulse':
            amount = noise_params.get('impulse_amount', 0.05)
            num_impulses = int(len(audio) * amount)
            impulse_indices = np.random.choice(len(audio), num_impulses, replace=False)
            noisy_audio[impulse_indices] = np.random.uniform(-1, 1, num_impulses)
    
    # Clip to prevent signal overflow
    noisy_audio = np.clip(noisy_audio, -1, 1)
    
    return noisy_audio

def prepare_audio_patches(audio_clips, patch_size=16000, stride=8000, max_patches_per_clip=50):
    """
    Extract patches from audio clips
    
    Args:
        audio_clips: List of clean audio numpy arrays
        patch_size: Size of audio patches
        stride: Stride between patches
        max_patches_per_clip: Maximum patches per audio clip
    
    Returns:
        Lists of clean patches for train/val/test
    """
    clean_patches = []
    
    for clip in tqdm(audio_clips, desc="Extracting Audio Patches"):
        patches_from_clip = 0
        
        for i in range(0, len(clip) - patch_size + 1, stride):
            patch = clip[i:i+patch_size]
            
            # Skip patches with very low energy
            if np.mean(np.abs(patch)) < 0.01:
                continue
            
            clean_patches.append(patch)
            patches_from_clip += 1
            
            if patches_from_clip >= max_patches_per_clip:
                break
    
    # Shuffle patches
    random.shuffle(clean_patches)
    
    # Split into train/val/test
    val_split = int(len(clean_patches) * 0.15)
    test_split = int(len(clean_patches) * 0.05)
    
    train_patches = clean_patches[val_split + test_split:]
    val_patches = clean_patches[:val_split]
    test_patches = clean_patches[val_split:val_split + test_split]
    
    print(f"Train: {len(train_patches)}, Validation: {len(val_patches)}, Test: {len(test_patches)} patches")
    
    return train_patches, val_patches, test_patches

class AudioDenoiseDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, clean_patches, batch_size=32, noise_types=None, noise_params=None, is_training=True):
        self.clean_patches = clean_patches
        self.batch_size = batch_size
        self.noise_types = noise_types
        self.noise_params = noise_params
        self.is_training = is_training
    
    def __len__(self):
        return len(self.clean_patches) // self.batch_size
    
    def __getitem__(self, idx):
        batch_clean = self.clean_patches[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_clean = np.array(batch_clean).reshape(-1, len(batch_clean[0]), 1)
        
        batch_noisy = []
        for clean_audio in batch_clean[:, :, 0]:
            if self.is_training:
                # Random noise configuration during training
                noise_types = random.choice([
                    ['gaussian'], 
                    ['white'], 
                    ['pink'], 
                    ['impulse']
                ])
                
                noise_params = {
                    'gaussian_std': random.choice([0.01, 0.05, 0.1]),
                    'white_amplitude': random.choice([0.01, 0.05, 0.1]),
                    'impulse_amount': random.choice([0.03, 0.05, 0.07])
                }
            else:
                noise_types = self.noise_types
                noise_params = self.noise_params
            
            noisy_audio = apply_noise(clean_audio, noise_types, noise_params)
            batch_noisy.append(noisy_audio)
        
        batch_noisy = np.array(batch_noisy).reshape(-1, len(batch_noisy[0]), 1)
        return batch_noisy, batch_clean

def combined_audio_loss(y_true, y_pred):
    """
    Combined loss function for audio denoising
    
    Combines mean squared error and spectral convergence
    """
    # Mean Squared Error
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Spectral Convergence Loss
    def spectral_convergence(y_true, y_pred):
        # Compute Short-time Fourier Transform
        y_true_stft = tf.signal.stft(y_true[:, :, 0], frame_length=512, frame_step=256)
        y_pred_stft = tf.signal.stft(y_pred[:, :, 0], frame_length=512, frame_step=256)
        
        # Magnitude of STFT
        y_true_mag = tf.abs(y_true_stft)
        y_pred_mag = tf.abs(y_pred_stft)
        
        # Spectral Convergence
        num = tf.norm(y_true_mag - y_pred_mag, ord='fro')
        den = tf.norm(y_true_mag, ord='fro')
        
        return num / (den + 1e-8)
    
    spectral_loss = spectral_convergence(y_true, y_pred)
    
    # Combined loss
    return 0.5 * mse_loss + 0.5 * spectral_loss

def residual_block_1d(input_tensor, filters, kernel_size=3):
    """
    1D Residual block for audio processing
    """
    x = Conv1D(filters, kernel_size, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Add residual connection
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    
    return x

def attention_block_1d(input_tensor, filters):
    """
    1D Attention mechanism for audio processing
    """
    x = Conv1D(filters, 1, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Compute attention map
    attention = Conv1D(filters, 1, padding='same', activation='sigmoid')(x)
    
    # Apply attention
    return Multiply()([x, attention])

def build_audio_denoising_model(input_shape=(16000, 1)):
    """
    Build a 1D U-Net with attention and residual blocks for audio denoising
    """
    with strategy.scope():
        inputs = Input(shape=input_shape)
        
        # Encoder
        conv1 = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = residual_block_1d(conv1, 64)
        pool1 = MaxPooling1D(2)(conv1)
        
        conv2 = Conv1D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = residual_block_1d(conv2, 128)
        pool2 = MaxPooling1D(2)(conv2)
        
        conv3 = Conv1D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = residual_block_1d(conv3, 256)
        pool3 = MaxPooling1D(2)(conv3)
        
        # Middle with attention
        middle = Conv1D(512, 3, activation='relu', padding='same')(pool3)
        middle = attention_block_1d(middle, 512)
        middle = residual_block_1d(middle, 512)
        
        # Decoder with skip connections
        up3 = UpSampling1D(2)(middle)
        up3 = Conv1D(256, 3, activation='relu', padding='same')(up3)
        merge3 = Concatenate()([conv3, up3])
        deconv3 = Conv1D(256, 3, activation='relu', padding='same')(merge3)
        deconv3 = residual_block_1d(deconv3, 256)
        
        up2 = UpSampling1D(2)(deconv3)
        up2 = Conv1D(128, 3, activation='relu', padding='same')(up2)
        merge2 = Concatenate()([conv2, up2])
        deconv2 = Conv1D(128, 3, activation='relu', padding='same')(merge2)
        deconv2 = residual_block_1d(deconv2, 128)
        
        up1 = UpSampling1D(2)(deconv2)
        up1 = Conv1D(64, 3, activation='relu', padding='same')(up1)
        merge1 = Concatenate()([conv1, up1])
        deconv1 = Conv1D(64, 3, activation='relu', padding='same')(merge1)
        deconv1 = residual_block_1d(deconv1, 64)
        
        # Output
        output = Conv1D(1, 3, activation='tanh', padding='same')(deconv1)
        
        # Model
        model = Model(inputs=inputs, outputs=output)
        
        # Compile with the combined loss
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=combined_audio_loss,
            metrics=['mse']
        )
        
        return model

def evaluate_audio_model(model, test_patches, noise_types=None):
    """
    Evaluate model performance on test audio patches
    """
    if noise_types is None:
        noise_types = [
            ['gaussian', {'gaussian_std': 0.05}],
            ['white', {'white_amplitude': 0.05}],
            ['pink'],
            ['impulse', {'impulse_amount': 0.05}]
        ]
    
    results = {}
    
    for noise_config in noise_types:
        noise_type = noise_config[0]
        params = noise_config[1] if len(noise_config) > 1 else {}
        
        print(f"Evaluating on {noise_type} noise...")
        mse_values = []
        snr_values = []
        
        for clean_audio in tqdm(test_patches[:100]):
            # Apply noise
            noisy_audio = apply_noise(clean_audio, [noise_type], params)
            
            # Denoise
            clean_audio = clean_audio.reshape(1, -1, 1)
            noisy_audio = noisy_audio.reshape(1, -1, 1)
            
            denoised_audio = model.predict(noisy_audio)[0, :, 0]
            
            # Calculate MSE
            mse = np.mean((clean_audio[0, :, 0] - denoised_audio) ** 2)
            
            # Calculate Signal-to-Noise Ratio (SNR)
            signal_power = np.mean(clean_audio[0, :, 0] ** 2)
            noise_power = np.mean((clean_audio[0, :, 0] - denoised_audio) ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            mse_values.append(mse)
            snr_values.append(snr)
        
        results[noise_type] = {
            'avg_mse': np.mean(mse_values),
            'avg_snr': np.mean(snr_values)
        }
        
        print(f"  Average MSE: {results[noise_type]['avg_mse']:.4f}")
        print(f"  Average SNR: {results[noise_type]['avg_snr']:.2f} dB")
    
    return results

def visualize_audio_results(model, test_patches, num_samples=4, save_path=None):
    """
    Visualize audio denoising results
    """
    plt.figure(figsize=(15, 10))
    
    noise_types = ['gaussian', 'white', 'pink', 'impulse']
    indices = np.random.choice(len(test_patches), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        clean_audio = test_patches[idx]
        
        plt.subplot(num_samples, len(noise_types) + 1, i * (len(noise_types) + 1) + 1)
        plt.plot(clean_audio)
        plt.title(f"Clean Audio {i+1}")
        plt.ylim(-1, 1)
        
        for j, noise_type in enumerate(noise_types):
            # Apply noise
            noisy_audio = apply_noise(clean_audio, [noise_type])
            
            # Denoise
            clean_audio_shaped = clean_audio.reshape(1, -1, 1)
            noisy_audio_shaped = noisy_audio.reshape(1, -1, 1)
            denoised_audio = model.predict(noisy_audio_shaped)[0, :, 0]
            
            # Plot results
            plt.subplot(num_samples, len(noise_types) + 1, i * (len(noise_types) + 1) + j + 2)
            plt.plot(noisy_audio, label='Noisy', alpha=0.7)
            plt.plot(denoised_audio, label='Denoised', alpha=0.7)
            plt.title(f"{noise_type.capitalize()} Noise")
            plt.ylim(-1, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def main():
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    # Dataset Path - This will be replaced with your actual input path
    DATASET_PATH = INPUT_DIR / 'vctk-corpus'  # Placeholder path
    
    # Load audio files
    print("Loading audio files...")
    audio_clips = load_audio_files(DATASET_PATH, max_duration=10, sr=16000, limit=500)
    
    # Prepare audio patches
    train_patches, val_patches, test_patches = prepare_audio_patches(
        audio_clips, 
        patch_size=16000, 
        stride=8000, 
        max_patches_per_clip=50
    )
    
    # Build model
    model = build_audio_denoising_model(input_shape=(16000, 1))
    model.summary()
    
    # Create data generators
    train_gen = AudioDenoiseDataGenerator(train_patches, batch_size=32)
    val_gen = AudioDenoiseDataGenerator(
        val_patches, 
        batch_size=32, 
        is_training=False,
        noise_types=[['gaussian'], ['white']]
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            str(MODEL_DIR / 'audio_denoising_model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras'),
            monitor='val_loss',
            save_best_only=True
        ),
        TensorBoard(log_dir=str(LOG_DIR)),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6
        ),
        EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
    ]
    
    # Training
    with strategy.scope():
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=100,
            callbacks=callbacks
        )
    
    # Evaluate model
    test_results = evaluate_audio_model(model, test_patches)
    
    # Visualize results
    save_path = WORKING_DIR / 'audio_denoising_results.png'
    visualize_audio_results(model, test_patches, save_path=save_path)
    
    # Save models
    model.save(WORKING_DIR / 'audio_denoising_model.keras')
    model.save(WORKING_DIR / 'audio_denoising_model.h5')
    
    # Save evaluation results
    with open(WORKING_DIR / 'audio_denoising_evaluation.json', 'w') as f:
        json.dump(test_results, f, indent=4)

if __name__ == '__main__':
    main()