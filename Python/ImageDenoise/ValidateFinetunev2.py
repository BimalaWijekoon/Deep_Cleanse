import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request
import json
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Custom loss function (must be defined to load the model)
def combined_loss(y_true, y_pred):
    """
    Combined loss function used during model training
    Combines MSE and SSIM losses
    """
    # MSE Loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # SSIM Loss (1-SSIM since we want to minimize)
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    
    # Combined loss (you can adjust the weights)
    return 0.5 * mse_loss + 0.5 * ssim_loss

# Noise application function (matching the original fine-tuning script)
def apply_noise(image, noise_types=None, noise_params=None):
    """
    Apply different types of noise to an image
    
    Args:
        image: Normalized image array [0,1]
        noise_types: List of noise types to apply ('gaussian', 'salt_pepper', 'speckle')
        noise_params: Parameters for the noise functions
    
    Returns:
        Noisy image array [0,1]
    """
    import random
    
    if noise_types is None:
        noise_types = random.choice([
            ['gaussian'], 
            ['salt_pepper'], 
            ['speckle'], 
            ['gaussian', 'salt_pepper'],
            ['gaussian', 'speckle']
        ])
    
    if noise_params is None:
        noise_params = {}
    
    # Make a copy to avoid modifying the original
    noisy_image = image.copy()
    
    # Apply each noise type in sequence
    for noise_type in noise_types:
        if noise_type == 'gaussian':
            # Randomize noise level if not specified
            std = noise_params.get('gaussian_std', 
                                   random.choice([0.02, 0.05, 0.1, 0.15, 0.2]))
            mean = noise_params.get('mean', 0)
            
            # Generate Gaussian noise
            noise = np.random.normal(mean, std, image.shape)
            noisy_image = noisy_image + noise
        
        elif noise_type == 'salt_pepper':
            # Randomize noise level if not specified
            amount = noise_params.get('salt_pepper_amount', 
                                     random.choice([0.01, 0.03, 0.05, 0.07]))
            
            # Generate salt and pepper noise
            salt_vs_pepper = random.uniform(0.3, 0.7)  # Varying salt/pepper ratio
            
            # Salt (white) noise
            num_salt = np.ceil(amount * image.size * salt_vs_pepper)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            noisy_image[tuple(coords)] = 1
            
            # Pepper (black) noise
            num_pepper = np.ceil(amount * image.size * (1 - salt_vs_pepper))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            noisy_image[tuple(coords)] = 0
        
        elif noise_type == 'speckle':
            # Randomize noise level if not specified
            var = noise_params.get('speckle_var', 
                                   random.choice([0.05, 0.1, 0.15]))
            
            # Generate speckle noise
            noise = np.random.randn(*image.shape) * var
            noisy_image = noisy_image + noisy_image * noise
    
    # Clip values to valid range [0,1]
    noisy_image = np.clip(noisy_image, 0.0, 1.0)
    
    return noisy_image

def download_kodak_dataset(output_dir):
    """
    Download Kodak dataset images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(1, 25):  # 24 Kodak images
        url = f"http://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png"
        output_path = output_dir / f"kodim{i:02d}.png"
        
        if not output_path.exists():
            print(f"Downloading {output_path.name}...")
            try:
                urllib.request.urlretrieve(url, output_path)
            except Exception as e:
                print(f"Failed to download {url}: {e}")

def load_and_validate_model(model_path):
    """
    Load the fine-tuned model with custom loss function
    """
    custom_objects = {
        'combined_loss': combined_loss
    }
    
    return load_model(model_path, custom_objects=custom_objects)

def validate_model(model, test_data_dir, results_dir):
    """
    Validate the model on different noise types and save visualizations
    """
    # Noise types to evaluate
    noise_types = [
        ('gaussian_low', ['gaussian'], {'gaussian_std': 0.5}),
        ('gaussian_high', ['gaussian'], {'gaussian_std': 0.5}),
        ('salt_pepper', ['salt_pepper'], {'salt_pepper_amount': 0.5}),
        ('speckle', ['speckle'], {'speckle_var': 0.5})
    ]
    
    results = {}
    
    # Ensure results directory exists
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test images
    test_images = []
    for img_path in Path(test_data_dir).glob('*.png'):
        img = np.array(Image.open(img_path).convert('RGB')) / 255.0
        test_images.append((img_path.stem, img))
    
    # Open a text file to log results
    log_path = results_dir / 'validation_log.txt'
    with open(log_path, 'w') as log_file:
        # Evaluate on each noise type
        for noise_name, noise_type, noise_params in noise_types:
            psnr_values = []
            ssim_values = []
            psnr_improvements = []
            ssim_improvements = []
            processing_times = []
            
            # Create a directory for this noise type
            noise_dir = results_dir / noise_name
            noise_dir.mkdir(parents=True, exist_ok=True)
            
            log_file.write(f"Evaluating {noise_name} noise...\n")
            for img_name, clean_img in tqdm(test_images):
                # Ensure image is tiled into 64x64 patches
                h, w = clean_img.shape[:2]
                
                # Initialize containers for denoised patches
                denoised_patches = []
                
                # Tile the image into 64x64 patches
                for y in range(0, h, 64):
                    for x in range(0, w, 64):
                        # Extract 64x64 patch
                        patch = clean_img[y:y+64, x:x+64]
                        
                        # If patch is not 64x64, pad it
                        if patch.shape[:2] != (64, 64):
                            padded_patch = np.zeros((64, 64, 3), dtype=clean_img.dtype)
                            padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                            patch = padded_patch
                        
                        # Apply noise
                        noisy_patch = apply_noise(patch, noise_type, noise_params)
                        
                        # Denoise patch
                        import time
                        start_time = time.time()
                        denoised_patch = model.predict(np.expand_dims(noisy_patch, axis=0))[0]
                        processing_time = time.time() - start_time
                        
                        denoised_patches.append((y, x, denoised_patch))
                        processing_times.append(processing_time)
                
                # Reconstruct the full image from denoised patches
                denoised_img = np.zeros_like(clean_img)
                for y, x, patch in denoised_patches:
                    denoised_img[y:y+64, x:x+64] = patch[:clean_img.shape[0]-y, :clean_img.shape[1]-x]
                
                # Calculate metrics
                clean_psnr = psnr(clean_img, denoised_img, data_range=1.0)
                clean_ssim = ssim(clean_img, denoised_img, data_range=1.0, channel_axis=2)
                
                noisy_img = apply_noise(clean_img, noise_type, noise_params)
                noisy_psnr = psnr(clean_img, noisy_img, data_range=1.0)
                noisy_ssim = ssim(clean_img, noisy_img, data_range=1.0, channel_axis=2)
                
                psnr_values.append(clean_psnr)
                ssim_values.append(clean_ssim)
                psnr_improvements.append(clean_psnr - noisy_psnr)
                ssim_improvements.append(clean_ssim - noisy_ssim)
                
                # Save images
                plt.figure(figsize=(15,5))
                plt.subplot(131)
                plt.title('Original Image')
                plt.imshow(clean_img)
                plt.axis('off')
                
                plt.subplot(132)
                plt.title(f'Noisy Image ({noise_name})')
                plt.imshow(noisy_img)
                plt.axis('off')
                
                plt.subplot(133)
                plt.title('Denoised Image')
                plt.imshow(denoised_img)
                plt.axis('off')
                
                # Save the figure
                plt.tight_layout()
                output_path = noise_dir / f'{img_name}_comparison.png'
                plt.savefig(output_path)
                plt.close()
            
            # Calculate average metrics
            results[noise_name] = {
                'psnr': np.mean(psnr_values),
                'ssim': np.mean(ssim_values),
                'psnr_improvement': np.mean(psnr_improvements),
                'ssim_improvement': np.mean(ssim_improvements),
                'processing_time': np.mean(processing_times)
            }
            
            # Log results for each noise type
            log_file.write(f"\n{noise_name.upper()}:\n")
            log_file.write(f"  Average PSNR: {results[noise_name]['psnr']:.2f} dB\n")
            log_file.write(f"  Average SSIM: {results[noise_name]['ssim']:.4f}\n")
            log_file.write(f"  Average PSNR Improvement: {results[noise_name]['psnr_improvement']:.2f} dB\n")
            log_file.write(f"  Average SSIM Improvement: {results[noise_name]['ssim_improvement']:.4f}\n")
            log_file.write(f"  Average Processing Time: {results[noise_name]['processing_time']:.4f} seconds\n")
        
        # Calculate overall performance
        results['overall'] = {
            'psnr': np.mean([r['psnr'] for r in results.values() if isinstance(r, dict)]),
            'ssim': np.mean([r['ssim'] for r in results.values() if isinstance(r, dict)]),
            'psnr_improvement': np.mean([r['psnr_improvement'] for r in results.values() if isinstance(r, dict)]),
            'ssim_improvement': np.mean([r['ssim_improvement'] for r in results.values() if isinstance(r, dict)])
        }
        
        # Log overall performance
        log_file.write("\nOVERALL PERFORMANCE:\n")
        log_file.write(f"  Average PSNR: {results['overall']['psnr']:.2f} dB\n")
        log_file.write(f"  Average SSIM: {results['overall']['ssim']:.4f}\n")
        log_file.write(f"  Average PSNR Improvement: {results['overall']['psnr_improvement']:.2f} dB\n")
        log_file.write(f"  Average SSIM Improvement: {results['overall']['ssim_improvement']:.4f}\n")
        
        # Log comparison with original validation report
        log_file.write("\nCOMPARISON WITH ORIGINAL VALIDATION REPORT:\n")
        
        # Original validation report values
        original_report = {
            'gaussian_low': {
                'psnr': 18.70,
                'ssim': 0.5134,
                'psnr_improvement': 7.41,
                'ssim_improvement': 0.1662
            },
            'gaussian_high': {
                'psnr': 18.59,
                'ssim': 0.4846,
                'psnr_improvement': 1.72,
                'ssim_improvement': 0.1572
            },
            'salt_pepper': {
                'psnr': 18.58,
                'ssim': 0.4885,
                'psnr_improvement': 0.04,
                'ssim_improvement': -0.0957
            },
            'speckle': {
                'psnr': 18.65,
                'ssim': 0.5134,
                'psnr_improvement': -8.40,
                'ssim_improvement': -0.2357
            }
        }
        
        log_file.write("\nNoise Type Comparison:\n")
        for noise_type in ['gaussian_low', 'gaussian_high', 'salt_pepper', 'speckle']:
            log_file.write(f"\n{noise_type.upper()}:\n")
            orig = original_report[noise_type]
            curr = results[noise_type]
            
            log_file.write(f"  Original PSNR: {orig['psnr']:.2f} | Current PSNR: {curr['psnr']:.2f}\n")
            log_file.write(f"  Original SSIM: {orig['ssim']:.4f} | Current SSIM: {curr['ssim']:.4f}\n")
            log_file.write(f"  Original PSNR Improvement: {orig['psnr_improvement']:.2f} | Current PSNR Improvement: {curr['psnr_improvement']:.2f}\n")
            log_file.write(f"  Original SSIM Improvement: {orig['ssim_improvement']:.4f} | Current SSIM Improvement: {curr['ssim_improvement']:.4f}\n")
    
    return results

def main():
    # Define paths
    MODEL_DIR = Path('FineTunedModelsv2')
    RESULTS_DIR = Path('validation_results_finetunedv2')
    TEST_DATA_DIR = Path('test_data_finetunedv2')

    # Create necessary directories
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    TEST_DATA_DIR.mkdir(exist_ok=True, parents=True)

    # Download Kodak dataset if not exists
    download_kodak_dataset(TEST_DATA_DIR)

    # Find the latest fine-tuned model
    model_paths = list(MODEL_DIR.glob('*.keras')) + list(MODEL_DIR.glob('*.h5'))
    if not model_paths:
        print("No model found in the directory.")
        return
    
    # Use the most recently modified model
    model_path = max(model_paths, key=os.path.getmtime)
    print(f"Using model: {model_path}")

    # Load the model
    model = load_and_validate_model(model_path)

    # Validate the model and save results
    results = validate_model(model, TEST_DATA_DIR, RESULTS_DIR)

    # Save results to JSON
    results_path = RESULTS_DIR / 'validation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Validation results saved to {results_path}")

if __name__ == '__main__':
    main()