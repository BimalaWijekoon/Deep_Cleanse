import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from sklearn.metrics import mean_squared_error
# Import PSNR from skimage instead of sklearn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import glob
import time
import cv2
import random

# Define paths
MODEL_DIR = Path('BaseModels')
RESULTS_DIR = Path('validation_results_base')
TEST_DATA_DIR = Path('test_data_base')

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

def download_test_images():
    """Download test images from different sources"""
    print("Downloading test images...")
    
    # Option 1: BSD68 dataset (common benchmark for denoising)
    bsd68_url = "https://webdav.tuebingen.mpg.de/pixel/benchmark/test/kodak/*.png"
    
    # Since direct downloading from the pattern isn't possible, let's use Kodak dataset instead
    kodak_url = "http://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
    
    # Download 10 images from Kodak dataset
    for i in range(1, 11):
        img_url = kodak_url.format(i)
        img_path = TEST_DATA_DIR / f"kodak_{i}.png"
        
        if not img_path.exists():
            try:
                urllib.request.urlretrieve(img_url, img_path)
                print(f"Downloaded {img_url} to {img_path}")
            except Exception as e:
                print(f"Failed to download {img_url}: {e}")
    
    # Option 2: If Kodak fails, use placeholder images
    if not list(TEST_DATA_DIR.glob("*.png")):
        print("Creating placeholder test images since download failed...")
        # Create some placeholder images
        for i in range(1, 6):
            img = np.ones((256, 256, 3)) * (i / 10.0)
            img_path = TEST_DATA_DIR / f"placeholder_{i}.png"
            plt.imsave(str(img_path), img)
    
    # Count downloaded images
    image_files = list(TEST_DATA_DIR.glob("*.png"))
    if not image_files:
        raise ValueError("No test images available. Please check your internet connection.")
    
    print(f"Successfully prepared {len(image_files)} test images")
    return image_files

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    """Load and preprocess a single image"""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    return img_array

def apply_noise(image, noise_type='gaussian', noise_params=None):
    """
    Apply different types of noise to an image
    
    Args:
        image: Normalized image array [0,1]
        noise_type: Type of noise to apply ('gaussian', 'salt_pepper', 'poisson', 'speckle')
        noise_params: Parameters for the noise function
    
    Returns:
        Noisy image array [0,1]
    """
    if noise_params is None:
        noise_params = {}
    
    # Make a copy to avoid modifying the original
    noisy_image = image.copy()
    
    if noise_type == 'gaussian':
        # Default parameters
        std = noise_params.get('std', 0.1)
        mean = noise_params.get('mean', 0)
        
        # Generate Gaussian noise
        noise = np.random.normal(mean, std, image.shape)
        noisy_image = image + noise
    
    elif noise_type == 'salt_pepper':
        # Default parameters
        amount = noise_params.get('amount', 0.05)
        
        # Generate salt and pepper noise
        salt_vs_pepper = 0.5  # Equal amounts of salt and pepper
        
        # Salt (white) noise
        num_salt = np.ceil(amount * image.size * salt_vs_pepper)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[tuple(coords)] = 1
        
        # Pepper (black) noise
        num_pepper = np.ceil(amount * image.size * (1 - salt_vs_pepper))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[tuple(coords)] = 0
    
    elif noise_type == 'poisson':
        # Poisson noise (simulates photon counting noise in images)
        # Scale image to appropriate values for Poisson noise
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        
        # Apply Poisson noise
        noisy_image = np.random.poisson(image * vals) / float(vals)
    
    elif noise_type == 'speckle':
        # Speckle noise (multiplicative noise)
        # Default parameters
        var = noise_params.get('var', 0.1)
        
        # Generate speckle noise
        noise = np.random.randn(*image.shape) * var
        noisy_image = image + image * noise
    
    # Clip values to valid range [0,1]
    noisy_image = np.clip(noisy_image, 0.0, 1.0)
    
    return noisy_image

def calculate_metrics(original, denoised, noisy=None):
    """
    Calculate performance metrics for the denoised image
    
    Args:
        original: Original clean image
        denoised: Denoised image
        noisy: Noisy image (optional)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Mean Squared Error (MSE) - lower is better
    metrics['mse'] = mean_squared_error(original.flatten(), denoised.flatten())
    
    # Peak Signal-to-Noise Ratio (PSNR) - higher is better
    metrics['psnr'] = peak_signal_noise_ratio(original, denoised, data_range=1.0)
    
    # Structural Similarity Index (SSIM) - higher is better
    # Fix the 'multichannel' parameter which is deprecated
    metrics['ssim'] = ssim(original, denoised, data_range=1.0, channel_axis=2)
    
    # If noisy image is provided, calculate improvement metrics
    if noisy is not None:
        # Metrics for noisy image
        metrics['noisy_mse'] = mean_squared_error(original.flatten(), noisy.flatten())
        metrics['noisy_psnr'] = peak_signal_noise_ratio(original, noisy, data_range=1.0)
        metrics['noisy_ssim'] = ssim(original, noisy, data_range=1.0, channel_axis=2)
        
        # Improvement metrics
        metrics['mse_improvement'] = metrics['noisy_mse'] - metrics['mse']
        metrics['psnr_improvement'] = metrics['psnr'] - metrics['noisy_psnr']
        metrics['ssim_improvement'] = metrics['ssim'] - metrics['noisy_ssim']
    
    return metrics

def visualize_results(original, noisy, denoised, metrics, save_path):
    """
    Create visualization of original, noisy, and denoised images with metrics
    
    Args:
        original: Original clean image
        noisy: Noisy image
        denoised: Denoised image
        metrics: Dictionary of metrics
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Display images
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(noisy)
    axes[1].set_title(f'Noisy\nPSNR: {metrics["noisy_psnr"]:.2f}, SSIM: {metrics["noisy_ssim"]:.4f}')
    axes[1].axis('off')
    
    axes[2].imshow(denoised)
    axes[2].set_title(f'Denoised\nPSNR: {metrics["psnr"]:.2f}, SSIM: {metrics["ssim"]:.4f}')
    axes[2].axis('off')
    
    # Add overall metrics as text
    plt.figtext(0.5, 0.01, 
                f'Improvements: PSNR: +{metrics["psnr_improvement"]:.2f} dB, SSIM: +{metrics["ssim_improvement"]:.4f}', 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_performance_by_noise_type(all_metrics, save_path):
    """Create bar charts comparing performance across noise types"""
    noise_types = list(all_metrics.keys())
    
    # Prepare data for plotting
    psnr_values = [np.mean([m['psnr'] for m in all_metrics[noise_type]]) for noise_type in noise_types]
    ssim_values = [np.mean([m['ssim'] for m in all_metrics[noise_type]]) for noise_type in noise_types]
    psnr_improvement = [np.mean([m['psnr_improvement'] for m in all_metrics[noise_type]]) for noise_type in noise_types]
    ssim_improvement = [np.mean([m['ssim_improvement'] for m in all_metrics[noise_type]]) for noise_type in noise_types]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot PSNR values
    axes[0, 0].bar(noise_types, psnr_values, color='blue')
    axes[0, 0].set_title('Average PSNR by Noise Type')
    axes[0, 0].set_ylabel('PSNR (dB)')
    axes[0, 0].set_ylim(bottom=0)
    
    # Plot SSIM values
    axes[0, 1].bar(noise_types, ssim_values, color='green')
    axes[0, 1].set_title('Average SSIM by Noise Type')
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].set_ylim(0, 1)
    
    # Plot PSNR improvement
    axes[1, 0].bar(noise_types, psnr_improvement, color='orange')
    axes[1, 0].set_title('Average PSNR Improvement by Noise Type')
    axes[1, 0].set_ylabel('PSNR Improvement (dB)')
    
    # Plot SSIM improvement
    axes[1, 1].bar(noise_types, ssim_improvement, color='purple')
    axes[1, 1].set_title('Average SSIM Improvement by Noise Type')
    axes[1, 1].set_ylabel('SSIM Improvement')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_best_and_worst(all_metrics, noise_types, image_paths, save_path):
    """
    Visualize the best and worst performing cases for each noise type
    
    Args:
        all_metrics: Dictionary of all metrics grouped by noise type
        noise_types: List of noise types
        image_paths: Dictionary of image paths by noise type and image index
        save_path: Path to save the visualization
    """
    # Number of rows = number of noise types
    # Columns: [Best PSNR, Worst PSNR]
    fig, axes = plt.subplots(len(noise_types), 2, figsize=(12, 4*len(noise_types)))
    
    for i, noise_type in enumerate(noise_types):
        # Get metrics for this noise type
        metrics = all_metrics[noise_type]
        
        # Find indices of best and worst cases based on PSNR
        psnr_values = [m['psnr'] for m in metrics]
        best_idx = np.argmax(psnr_values)
        worst_idx = np.argmin(psnr_values)
        
        # Load best and worst images
        best_paths = image_paths[noise_type][best_idx]
        worst_paths = image_paths[noise_type][worst_idx]
        
        best_original = plt.imread(best_paths['original'])
        best_noisy = plt.imread(best_paths['noisy'])
        best_denoised = plt.imread(best_paths['denoised'])
        
        worst_original = plt.imread(worst_paths['original'])
        worst_noisy = plt.imread(worst_paths['noisy'])
        worst_denoised = plt.imread(worst_paths['denoised'])
        
        # Create composite images (original/noisy/denoised side by side)
        best_composite = np.hstack([best_original, best_noisy, best_denoised])
        worst_composite = np.hstack([worst_original, worst_noisy, worst_denoised])
        
        # Handle single noise type case (prevents indexing error)
        if len(noise_types) == 1:
            ax_best = axes[0]
            ax_worst = axes[1]
        else:
            ax_best = axes[i, 0]
            ax_worst = axes[i, 1]
            
        # Plot best case
        ax_best.imshow(best_composite)
        ax_best.set_title(f'Best for {noise_type}\nPSNR: {metrics[best_idx]["psnr"]:.2f}')
        ax_best.axis('off')
        
        # Plot worst case
        ax_worst.imshow(worst_composite)
        ax_worst.set_title(f'Worst for {noise_type}\nPSNR: {metrics[worst_idx]["psnr"]:.2f}')
        ax_worst.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # 1. Load the saved model
    print("Loading the denoising model...")
    try:
        # Try loading .keras model (newest format)
        model_path = MODEL_DIR / "denoising_autoencoder_final.keras"
        if model_path.exists():
            model = load_model(model_path)
        else:
            # Try loading .h5 model (older format)
            model_path = MODEL_DIR / "denoising_autoencoder_final.h5"
            if model_path.exists():
                model = load_model(model_path)
            else:
                # Try any available model file
                model_files = list(MODEL_DIR.glob("*.h5")) + list(MODEL_DIR.glob("*.keras"))
                if model_files:
                    model_path = model_files[0]
                    model = load_model(model_path)
                else:
                    raise FileNotFoundError("No model files found in the imagemodels directory")
        
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Print model summary
    model.summary()
    
    # 2. Download test images
    try:
        image_files = download_test_images()
    except Exception as e:
        print(f"Error downloading test images: {e}")
        return
    
    # 3. Define noise types and parameters to test
    noise_types = {
        'gaussian_low': {'type': 'gaussian', 'params': {'std': 0.5}},
        'gaussian_high': {'type': 'gaussian', 'params': {'std': 0.5}},
        'salt_pepper': {'type': 'salt_pepper', 'params': {'amount': 0.5}},
        'speckle': {'type': 'speckle', 'params': {'var': 0.5}}
    }
    
    # 4. Process each image with each noise type
    all_metrics = {noise_name: [] for noise_name in noise_types.keys()}
    image_paths = {noise_name: [] for noise_name in noise_types.keys()}
    
    for img_idx, img_file in enumerate(image_files):
        print(f"Processing image {img_idx+1}/{len(image_files)}: {img_file.name}")
        
        # Load and preprocess original image
        original_img = load_and_preprocess_image(img_file)
        
        # Save original image
        original_path = ORIGINAL_DIR / f"original_{img_idx}.png"
        plt.imsave(str(original_path), original_img)
        
        # Process with each noise type
        for noise_name, noise_config in noise_types.items():
            print(f"  Applying {noise_name} noise...")
            
            # Apply noise
            noisy_img = apply_noise(
                original_img, 
                noise_type=noise_config['type'], 
                noise_params=noise_config['params']
            )
            
            # Save noisy image
            noisy_path = NOISY_DIR / f"{noise_name}_noisy_{img_idx}.png"
            plt.imsave(str(noisy_path), noisy_img)
            
            # Denoise image
            start_time = time.time()
            denoised_img = model.predict(np.expand_dims(noisy_img, axis=0))[0]
            processing_time = time.time() - start_time
            
            # Save denoised image
            denoised_path = DENOISED_DIR / f"{noise_name}_denoised_{img_idx}.png"
            plt.imsave(str(denoised_path), denoised_img)
            
            # Calculate metrics
            metrics = calculate_metrics(original_img, denoised_img, noisy_img)
            metrics['processing_time'] = processing_time
            all_metrics[noise_name].append(metrics)
            
            # Save paths for later visualization
            image_paths[noise_name].append({
                'original': str(original_path),
                'noisy': str(noisy_path),
                'denoised': str(denoised_path)
            })
            
            # Create individual result visualization
            viz_path = METRICS_DIR / f"{noise_name}_results_{img_idx}.png"
            visualize_results(original_img, noisy_img, denoised_img, metrics, viz_path)
    
    # 5. Generate summary visualizations and reports
    print("Generating summary visualizations and reports...")
    
    # Performance by noise type
    noise_comparison_path = METRICS_DIR / "noise_type_comparison.png"
    visualize_performance_by_noise_type(all_metrics, noise_comparison_path)
    
    # Best and worst case visualization
    best_worst_path = METRICS_DIR / "best_worst_cases.png"
    visualize_best_and_worst(all_metrics, list(noise_types.keys()), image_paths, best_worst_path)
    
    # Create summary report
    report_path = RESULTS_DIR / "validation_summary.txt"
    with open(report_path, 'w') as f:
        f.write("IMAGE DENOISING MODEL VALIDATION REPORT\n")
        f.write("=====================================\n\n")
        
        f.write("SUMMARY BY NOISE TYPE\n")
        f.write("-----------------\n")
        for noise_type in noise_types.keys():
            metrics = all_metrics[noise_type]
            avg_psnr = np.mean([m['psnr'] for m in metrics])
            avg_ssim = np.mean([m['ssim'] for m in metrics])
            avg_psnr_improvement = np.mean([m['psnr_improvement'] for m in metrics])
            avg_ssim_improvement = np.mean([m['ssim_improvement'] for m in metrics])
            avg_time = np.mean([m['processing_time'] for m in metrics])
            
            f.write(f"\n{noise_type.upper()}:\n")
            f.write(f"  Average PSNR: {avg_psnr:.2f} dB\n")
            f.write(f"  Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"  Average PSNR Improvement: +{avg_psnr_improvement:.2f} dB\n")
            f.write(f"  Average SSIM Improvement: +{avg_ssim_improvement:.4f}\n")
            f.write(f"  Average Processing Time: {avg_time:.4f} seconds\n")
        
        # Overall statistics
        all_psnr = [m['psnr'] for noise_type in all_metrics.keys() for m in all_metrics[noise_type]]
        all_ssim = [m['ssim'] for noise_type in all_metrics.keys() for m in all_metrics[noise_type]]
        all_psnr_improvement = [m['psnr_improvement'] for noise_type in all_metrics.keys() for m in all_metrics[noise_type]]
        all_ssim_improvement = [m['ssim_improvement'] for noise_type in all_metrics.keys() for m in all_metrics[noise_type]]
        
        f.write("\nOVERALL PERFORMANCE:\n")
        f.write("------------------\n")
        f.write(f"  Average PSNR: {np.mean(all_psnr):.2f} dB\n")
        f.write(f"  Average SSIM: {np.mean(all_ssim):.4f}\n")
        f.write(f"  Average PSNR Improvement: +{np.mean(all_psnr_improvement):.2f} dB\n")
        f.write(f"  Average SSIM Improvement: +{np.mean(all_ssim_improvement):.4f}\n")
        
        f.write("\nAREAS FOR IMPROVEMENT:\n")
        f.write("---------------------\n")
        
        # Find the worst performing noise type
        avg_psnr_by_noise = {noise_type: np.mean([m['psnr'] for m in all_metrics[noise_type]]) 
                             for noise_type in noise_types.keys()}
        worst_noise_type = min(avg_psnr_by_noise, key=avg_psnr_by_noise.get)
        
        f.write(f"1. The model performs worst on {worst_noise_type} noise.\n")
        f.write(f"   Consider fine-tuning specifically for this noise type.\n")
        
        # Check for patterns in worst cases
        worst_cases = []
        for noise_type in noise_types.keys():
            metrics = all_metrics[noise_type]
            worst_idx = np.argmin([m['psnr'] for m in metrics])
            worst_cases.append((noise_type, worst_idx))
        
        f.write("\n2. Common characteristics in worst-performing images:\n")
        # This would typically involve more sophisticated analysis of image characteristics
        f.write("   - Consider fine-tuning on images with high-frequency details or textures\n")
        f.write("   - Test more augmentation techniques during training\n")
        f.write("   - Consider adaptive noise level estimation to handle varying noise levels\n")
        
        f.write("\nRECOMMENDATIONS:\n")
        f.write("---------------\n")
        f.write("1. Fine-tune the model with mixed noise types for better generalization\n")
        f.write("2. Consider increasing the depth or capacity of the model\n")
        f.write("3. Experiment with attention mechanisms to better handle complex textures\n")
        f.write("4. Implement a noise level estimation module for adaptive denoising\n")
    
    print(f"Validation complete! Results saved to {RESULTS_DIR}")
    print(f"Summary report saved to {report_path}")

if __name__ == "__main__":
    main()