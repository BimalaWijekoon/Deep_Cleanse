import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import librosa
import soundfile as sf
from pathlib import Path
import urllib.request
import tarfile
import random
import time
import json

# Configuration for finetuned model
BASE_DIR = Path('.')
MODEL_DIR = BASE_DIR / 'FineTunedModels'
RESULTS_DIR = Path('validation_results_finetuned')
TEST_DATA_DIR = Path('test_data_finetuned')

# Create necessary directories
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
TEST_DATA_DIR.mkdir(exist_ok=True, parents=True)

# Create subdirectories for different types of results
ORIGINAL_DIR = RESULTS_DIR / 'original'
NOISY_DIR = RESULTS_DIR / 'noisy'
DENOISED_DIR = RESULTS_DIR / 'denoised'
METRICS_DIR = RESULTS_DIR / 'metrics'

for directory in [ORIGINAL_DIR, NOISY_DIR, DENOISED_DIR, METRICS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

def combined_audio_loss(y_true, y_pred):
    """
    Combined loss function for audio denoising (copied from FinetuneBase.py)
    
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

def load_custom_model(model_path):
    """
    Load custom model with the specific loss function
    """
    custom_objects = {
        'combined_audio_loss': combined_audio_loss
    }
    
    return load_model(model_path, custom_objects=custom_objects)

def download_test_audio():
    """
    Download test audio files if no existing dataset is found
    
    Returns:
        List of audio file paths
    """
    print("Checking for existing test audio files...")
    
    # First, check if there are already FLAC files in the directory
    existing_wav_files = list(TEST_DATA_DIR.rglob("*.flac"))
    
    if existing_wav_files:
        print(f"Found {len(existing_wav_files)} existing audio files. Using existing dataset.")
        return existing_wav_files[:20]
    
    # Check if tar file exists
    librispeech_url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    librispeech_file = TEST_DATA_DIR / "dev-clean.tar.gz"
    
    # If tar file exists, but no FLAC files extracted, extract it
    if librispeech_file.exists():
        print("Tar file found. Extracting dataset...")
        try:
            with tarfile.open(librispeech_file, 'r:gz') as tar_ref:
                tar_ref.extractall(TEST_DATA_DIR)
            print("Dataset extracted successfully!")
        except Exception as e:
            print(f"Error extracting dataset: {e}")
            return []
    else:
        # If tar file doesn't exist, download it
        print(f"Downloading LibriSpeech dataset from {librispeech_url}...")
        try:
            urllib.request.urlretrieve(librispeech_url, librispeech_file)
            
            # Extract the archive
            with tarfile.open(librispeech_file, 'r:gz') as tar_ref:
                tar_ref.extractall(TEST_DATA_DIR)
            print("Dataset downloaded and extracted successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return []
    
    # Find wav files after download/extraction
    wav_files = list(TEST_DATA_DIR.rglob("*.flac"))
    
    if not wav_files:
        raise ValueError("No audio files found. Check download and extraction.")
    
    # Limit to first 20 files
    return wav_files[:20]

def apply_noise(audio, noise_type='gaussian', noise_params=None):
    """
    Apply different types of noise to an audio signal
    """
    if noise_params is None:
        noise_params = {}
    
    # Copy audio to avoid modifying original
    noisy_audio = audio.copy()
    
    if noise_type == 'gaussian':
        # Default parameters
        std = noise_params.get('std', 0.05)
        mean = noise_params.get('mean', 0)
        
        # Generate Gaussian noise
        noise = np.random.normal(mean, std, audio.shape)
        noisy_audio = audio + noise
    
    elif noise_type == 'white':
        # White noise with variable intensity
        intensity = noise_params.get('intensity', 0.1)
        noise = np.random.normal(0, intensity, audio.shape)
        noisy_audio = audio + noise
    
    elif noise_type == 'impulse':
        # Impulse noise (random spikes)
        num_impulses = int(len(audio) * noise_params.get('rate', 0.01))
        impulse_indices = np.random.choice(len(audio), num_impulses, replace=False)
        noisy_audio[impulse_indices] = np.random.uniform(-1, 1, num_impulses)
    
    # Normalize back to original scale
    max_val = np.max(np.abs(audio))
    noisy_audio = noisy_audio * (max_val / np.max(np.abs(noisy_audio)))
    
    return noisy_audio

def compute_audio_metrics(original, denoised, sample_rate):
    """
    Compute audio quality metrics
    """
    metrics = {}
    
    # Ensure original and denoised have the same length
    min_length = min(len(original), len(denoised))
    original = original[:min_length]
    denoised = denoised[:min_length]
    
    # Signal-to-Noise Ratio (SNR)
    def calculate_snr(signal, noise):
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)
        return 10 * np.log10(signal_power / noise_power)
    
    # Mean Squared Error
    metrics['mse'] = np.mean((original - denoised)**2)
    
    # Signal-to-Noise Ratio
    metrics['snr'] = calculate_snr(original, original - denoised)
    
    # Perceptual Evaluation of Speech Quality (PESQ) - Optional
    try:
        from pypesq import pesq
        metrics['pesq'] = pesq(original, denoised, sample_rate)
    except ImportError:
        metrics['pesq'] = None
    
    # Short-Time Fourier Transform (STFT) based metrics
    original_spec = np.abs(librosa.stft(original))
    denoised_spec = np.abs(librosa.stft(denoised))
    
    # Spectral convergence
    metrics['spectral_convergence'] = np.linalg.norm(original_spec - denoised_spec) / np.linalg.norm(original_spec)
    
    return metrics

def visualize_audio_results(original, noisy, denoised, metrics, save_path, sample_rate):
    """
    Create visualization of audio signals and their spectrograms
    """
    plt.figure(figsize=(15, 10))
    
    # Time domain plots
    plt.subplot(3, 2, 1)
    plt.plot(original)
    plt.title('Original Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 2, 2)
    plt.specgram(original, Fs=sample_rate)
    plt.title('Original Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    
    plt.subplot(3, 2, 3)
    plt.plot(noisy)
    plt.title('Noisy Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 2, 4)
    plt.specgram(noisy, Fs=sample_rate)
    plt.title('Noisy Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    
    plt.subplot(3, 2, 5)
    plt.plot(denoised)
    plt.title('Denoised Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 2, 6)
    plt.specgram(denoised, Fs=sample_rate)
    plt.title('Denoised Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    
    # Add metrics text
    metrics_text = "\n".join([f"{k}: {v}" for k, v in metrics.items() if v is not None])
    plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Find model files
    model_files = list(MODEL_DIR.glob('*.keras')) + list(MODEL_DIR.glob('*.h5'))
    
    if not model_files:
        print(f"No model files found in {MODEL_DIR}. Please check the directory.")
        return
    
    # Load the most recently created model
    model_path = max(model_files, key=os.path.getctime)
    
    print(f"Loading audio denoising model from {model_path}...")
    try:
        model = load_custom_model(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Download or prepare test audio files
    try:
        audio_files = download_test_audio()
        if not audio_files:
            print("No audio files available for testing.")
            return
    except Exception as e:
        print(f"Error preparing test audio: {e}")
        return
    
    # Noise types to test (aligned with the base script)
    noise_types = {
        'GAUSSIAN_LOW': {'type': 'gaussian', 'params': {'std': 0.05}},
        'GAUSSIAN_HIGH': {'type': 'gaussian', 'params': {'std': 0.15}},
        'WHITE_NOISE': {'type': 'white', 'params': {'intensity': 0.1}},
        'IMPULSE_NOISE': {'type': 'impulse', 'params': {'rate': 0.01}}
    }
    
    # Prepare results storage
    all_metrics = {noise_name: [] for noise_name in noise_types.keys()}
    
    # Process each audio file
    for idx, audio_file in enumerate(audio_files):
        print(f"Processing audio {idx+1}/{len(audio_files)}: {audio_file.name}")
        
        # Load audio with explicit 1-second duration at 16000 Hz
        original_audio, sample_rate = librosa.load(audio_file, sr=16000, duration=1)
        
        # Ensure input is exactly 16000 samples
        if len(original_audio) < 16000:
            # Pad with zeros if shorter
            original_audio = np.pad(original_audio, (0, 16000 - len(original_audio)), mode='constant')
        elif len(original_audio) > 16000:
            # Truncate if longer
            original_audio = original_audio[:16000]
        
        # Process with each noise type
        for noise_name, noise_config in noise_types.items():
            print(f"  Applying {noise_name} noise...")
            
            # Apply noise
            noisy_audio = apply_noise(
                original_audio, 
                noise_type=noise_config['type'], 
                noise_params=noise_config['params']
            )
            
            # Reshape for model input (ensure 16000 samples, single channel)
            noisy_input = noisy_audio.reshape(1, 16000, 1)
            original_input = original_audio.reshape(1, 16000, 1)
            
            # Denoise
            start_time = time.time()
            denoised_audio = model.predict(noisy_input)[0, :, 0]
            processing_time = time.time() - start_time
            
            # Compute metrics
            metrics = compute_audio_metrics(original_audio, denoised_audio, sample_rate)
            metrics['processing_time'] = processing_time
            all_metrics[noise_name].append(metrics)
            
            # Visualize results
            viz_path = METRICS_DIR / f"{noise_name}_results_{idx}.png"
            visualize_audio_results(
                original_audio, 
                noisy_audio, 
                denoised_audio, 
                metrics, 
                viz_path, 
                sample_rate
            )
            
            # Save audio files
            sf.write(ORIGINAL_DIR / f"original_{idx}_{noise_name}.wav", original_audio, sample_rate)
            sf.write(NOISY_DIR / f"noisy_{idx}_{noise_name}.wav", noisy_audio, sample_rate)
            sf.write(DENOISED_DIR / f"denoised_{idx}_{noise_name}.wav", denoised_audio, sample_rate)
    
    # Prepare comparison with base model validation results
    base_result_path = Path('validation_results_base/audio_validation_summary.txt')
    base_metrics = {}
    
    if base_result_path.exists():
        with open(base_result_path, 'r') as f:
            base_content = f.read()
    
    # Generate summary report
    report_path = RESULTS_DIR / "audio_validation_summary.txt"
    with open(report_path, 'w') as f:
        f.write("FINETUNED AUDIO DENOISING MODEL VALIDATION REPORT\n")
        f.write("===============================================\n\n")
        
        f.write("SUMMARY BY NOISE TYPE\n")
        f.write("-----------------\n")
        
        for noise_type in noise_types.keys():
            metrics = all_metrics[noise_type]
            
            # Compute averages
            avg_mse = np.mean([m['mse'] for m in metrics])
            avg_snr = np.mean([m['snr'] for m in metrics])
            avg_spectral_convergence = np.mean([m['spectral_convergence'] for m in metrics])
            avg_processing_time = np.mean([m['processing_time'] for m in metrics])
            
            f.write(f"\n{noise_type}:\n")
            f.write(f"  Average MSE: {avg_mse:.4f}\n")
            f.write(f"  Average SNR: {avg_snr:.2f} dB\n")
            f.write(f"  Average Spectral Convergence: {avg_spectral_convergence:.4f}\n")
            f.write(f"  Average Processing Time: {avg_processing_time:.4f} seconds\n")
        
        # Add comparison with base model if available
        if base_result_path.exists():
            f.write("\nCOMPARISON WITH BASE MODEL\n")
            f.write("------------------------\n")
            f.write(base_content)
        
        f.write("\nRECOMMENDATIONS:\n")
        f.write("---------------\n")
        f.write("1. Compare metrics with base model\n")
        f.write("2. Analyze performance improvements\n")
        f.write("3. Consider further fine-tuning if needed\n")
    
    print(f"Validation complete! Results saved to {RESULTS_DIR}")
    print(f"Summary report saved to {report_path}")

if __name__ == "__main__":
    main()