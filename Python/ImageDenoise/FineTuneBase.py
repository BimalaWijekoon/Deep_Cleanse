import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, SpatialDropout2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense, Reshape, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request
import zipfile
import tarfile
import time
import glob
import random
import cv2
from PIL import Image
from tqdm.notebook import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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
    # Kaggle directories - use /kaggle/working for all outputs
    BASE_DIR = Path('/kaggle')
    WORKING_DIR = BASE_DIR / 'working'
    MODEL_DIR = WORKING_DIR / 'models'
    LOG_DIR = WORKING_DIR / 'logs'
    INPUT_DIR = BASE_DIR / 'input'

# Create directories
WORKING_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Data directories
DATA_DIR = WORKING_DIR / 'data'
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'val'
TEST_DIR = DATA_DIR / 'test'

for directory in [DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

print("TensorFlow version:", tf.__version__)

# Check for GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU is available. Setting up mirrored strategy...")
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")
else:
    print("No GPU found. Using default strategy.")
    strategy = tf.distribute.get_strategy()

# Function to download and extract datasets
def download_datasets():
    """Download and extract multiple datasets for image denoising fine-tuning"""
    datasets = [
        {
            'name': 'BSD300',
            'url': 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz',
            'file_type': 'tgz',
            'extract_dir': DATA_DIR / 'BSD300'
        },
        {
            'name': 'DIV2K_valid',
            'url': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
            'file_type': 'zip',
            'extract_dir': DATA_DIR / 'DIV2K'
        },
        {
            'name': 'Kodak',
            'url': None,  # We'll download images individually
            'extract_dir': DATA_DIR / 'Kodak'
        }
    ]
    
    for dataset in datasets:
        os.makedirs(dataset['extract_dir'], exist_ok=True)
        
        # Special handling for Kodak dataset
        if dataset['name'] == 'Kodak':
            print("Downloading Kodak dataset...")
            for i in range(1, 25):  # Kodak dataset has 24 images
                try:
                    kodak_url = f"http://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png"
                    save_path = dataset['extract_dir'] / f"kodim{i:02d}.png"
                    
                    # Only download if file doesn't exist
                    if not save_path.exists():
                        urllib.request.urlretrieve(kodak_url, save_path)
                        print(f"Downloaded {save_path.name}")
                except Exception as e:
                    print(f"Failed to download Kodak image {i}: {e}")
            continue
        
        # Rest of the download logic remains the same for other datasets
        save_path = DATA_DIR / f"{dataset['name']}.{dataset['file_type']}"
        
        # Check if the dataset already exists
        if not os.path.exists(save_path):
            print(f"Downloading {dataset['name']} dataset...")
            try:
                urllib.request.urlretrieve(dataset['url'], save_path)
                print(f"Downloaded {dataset['name']} dataset")
            except Exception as e:
                print(f"Failed to download {dataset['name']} dataset: {e}")
                continue
        
        # Extract the dataset
        try:
            if dataset['file_type'] == 'zip':
                with zipfile.ZipFile(save_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset['extract_dir'])
            elif dataset['file_type'] == 'tgz':
                with tarfile.open(save_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(dataset['extract_dir'])
            print(f"Extracted {dataset['name']} dataset")
        except Exception as e:
            print(f"Failed to extract {dataset['name']} dataset: {e}")
            continue
# Alternatively, try to use existing Kaggle datasets if available
def find_kaggle_datasets():
    """Find and use existing image datasets in Kaggle input directory"""
    potential_dirs = [
        INPUT_DIR / 'div2k-high-resolution-images',
        INPUT_DIR / 'bsd300',
        INPUT_DIR / 'bsd68',
        INPUT_DIR / 'waterloo-exploration-database',
        INPUT_DIR / 'coco-2017',
        # Add more potential Kaggle dataset paths here
    ]
    
    found_datasets = []
    for dir_path in potential_dirs:
        if dir_path.exists():
            found_datasets.append(dir_path)
            print(f"Found dataset: {dir_path}")
    
    return found_datasets

# Function to apply various types of noise to images
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
    if noise_types is None:
        # If no noise types specified, randomly select one
        noise_types = random.choice([
            ['gaussian'], 
            ['salt_pepper'], 
            ['speckle'], 
            ['gaussian', 'salt_pepper'],  # Mixed noise types
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

# Data loading and preprocessing functions
def get_image_files(kaggle_datasets=None):
    """
    Get all image files from downloaded datasets or Kaggle input datasets
    
    Returns:
        List of image file paths
    """
    if kaggle_datasets:
        # Use Kaggle datasets if available
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        image_files = []
        
        for dataset_dir in kaggle_datasets:
            for ext in image_extensions:
                image_files.extend(list(Path(dataset_dir).glob(f'**/{ext}')))
        
        print(f"Found {len(image_files)} images in Kaggle datasets")
    else:
        # Use downloaded datasets
        image_files = []
        
        # BSD300 dataset
        bsd_dir = DATA_DIR / 'BSD300' / 'BSDS300' / 'images'
        if bsd_dir.exists():
            for subdir in ['train', 'test']:
                image_files.extend(list((bsd_dir / subdir).glob('*.jpg')))
        
        # DIV2K dataset
        div2k_dir = DATA_DIR / 'DIV2K'
        if div2k_dir.exists():
            image_files.extend(list(div2k_dir.glob('**/*.png')))
        
        # Kodak dataset
        kodak_dir = DATA_DIR / 'Kodak'
        if kodak_dir.exists():
            image_files.extend(list(kodak_dir.glob('*.png')))
            image_files.extend(list(kodak_dir.glob('*.jpg')))
        
        print(f"Found {len(image_files)} images in downloaded datasets")
    
    # Shuffle the files for better training
    random.shuffle(image_files)
    return image_files

def prepare_patches(image_files, patch_size=64, stride=32, max_patches_per_image=20, max_total_patches=30000):
    """
    Extract patches from images and split into train/val/test sets
    
    Args:
        image_files: List of image file paths
        patch_size: Size of patches to extract
        stride: Stride between patches
        max_patches_per_image: Maximum number of patches to extract per image
        max_total_patches: Maximum total number of patches to extract
        
    Returns:
        Lists of clean patches for train, val, test sets
    """
    clean_patches = []
    total_patches = 0
    
    print(f"Extracting patches from {len(image_files)} images...")
    for img_path in tqdm(image_files):
        try:
            # Load and normalize image
            img = np.array(Image.open(img_path).convert('RGB')) / 255.0
            
            # Skip images that are too small
            if img.shape[0] < patch_size or img.shape[1] < patch_size:
                continue
            
            # Extract patches
            patches_from_img = 0
            for i in range(0, img.shape[0] - patch_size + 1, stride):
                for j in range(0, img.shape[1] - patch_size + 1, stride):
                    patch = img[i:i+patch_size, j:j+patch_size]
                    
                    # Skip patches with low variance (flat regions)
                    if patch.std() < 0.03:
                        continue
                    
                    clean_patches.append(patch)
                    patches_from_img += 1
                    total_patches += 1
                    
                    if patches_from_img >= max_patches_per_image:
                        break
                
                if patches_from_img >= max_patches_per_image:
                    break
            
            if total_patches >= max_total_patches:
                break
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Shuffle patches
    random.shuffle(clean_patches)
    print(f"Extracted {len(clean_patches)} patches")
    
    # Split into train/val/test
    val_split = int(len(clean_patches) * 0.15)
    test_split = int(len(clean_patches) * 0.05)
    
    train_patches = clean_patches[val_split + test_split:]
    val_patches = clean_patches[:val_split]
    test_patches = clean_patches[val_split:val_split + test_split]
    
    print(f"Train: {len(train_patches)}, Validation: {len(val_patches)}, Test: {len(test_patches)} patches")
    return train_patches, val_patches, test_patches

# Data generator for training and validation
class DenoiseDataGenerator(tf.keras.utils.Sequence):
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
        batch_clean = np.array(batch_clean)
        
        # Apply various noise types to create noisy images
        batch_noisy = []
        for clean_img in batch_clean:
            # For each image, randomly select noise types and parameters
            if self.is_training:
                # During training, use random noise types and levels
                noise_types = random.choice([
                    ['gaussian'], 
                    ['salt_pepper'], 
                    ['speckle'], 
                    ['gaussian', 'salt_pepper'],
                    ['gaussian', 'speckle']
                ])
                
                # For gaussian noise, vary the standard deviation
                gaussian_std = random.choice([0.05, 0.1, 0.15])
                
                # For salt_pepper noise, vary the amount
                salt_pepper_amount = random.choice([0.03, 0.05, 0.07])
                
                # For speckle noise, vary the variance
                speckle_var = random.choice([0.05, 0.1, 0.15])
                
                noise_params = {
                    'gaussian_std': gaussian_std,
                    'salt_pepper_amount': salt_pepper_amount,
                    'speckle_var': speckle_var
                }
            else:
                # During validation, use fixed noise types if specified
                noise_types = self.noise_types
                noise_params = self.noise_params
            
            noisy_img = apply_noise(clean_img, noise_types, noise_params)
            batch_noisy.append(noisy_img)
        
        batch_noisy = np.array(batch_noisy)
        return batch_noisy, batch_clean

# Custom loss function combining MSE and SSIM
def combined_loss(y_true, y_pred):
    # MSE Loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # SSIM Loss (1-SSIM since we want to minimize)
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    
    # Combined loss (you can adjust the weights)
    return 0.5 * mse_loss + 0.5 * ssim_loss

# Model architecture improvements based on validation report recommendations
def attention_block(input_tensor, filters):
    """Attention mechanism to focus on relevant features"""
    x = Conv2D(filters, kernel_size=1, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Compute attention map
    attention = Conv2D(filters, kernel_size=1, padding='same')(x)
    attention = BatchNormalization()(attention)
    attention = Activation('sigmoid')(attention)
    
    # Apply attention
    return Multiply()([x, attention])

def residual_block(input_tensor, filters):
    """Residual block for better gradient flow"""
    x = Conv2D(filters, kernel_size=3, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Add residual connection
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    
    return x


def build_improved_model(input_shape=(64, 64, 3)):
    """
    Build an improved denoising model with attention mechanisms
    and deeper architecture as suggested in the validation report
    
    Args:
        input_shape: Fixed input shape for the model, default is (64, 64, 3)
    """
    with strategy.scope():
        # Input with a fixed shape
        inputs = Input(shape=input_shape)
        
        # Encoder
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = residual_block(conv1, 64)
        pool1 = MaxPooling2D((2, 2))(conv1)
        
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = residual_block(conv2, 128)
        pool2 = MaxPooling2D((2, 2))(conv2)
        
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = residual_block(conv3, 256)
        pool3 = MaxPooling2D((2, 2))(conv3)
        
        # Middle with attention
        middle = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
        middle = attention_block(middle, 512)
        middle = residual_block(middle, 512)
        
        # Decoder with skip connections (U-Net style)
        up3 = UpSampling2D((2, 2))(middle)
        up3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
        merge3 = Concatenate()([conv3, up3])
        deconv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge3)
        deconv3 = residual_block(deconv3, 256)
        
        up2 = UpSampling2D((2, 2))(deconv3)
        up2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
        merge2 = Concatenate()([conv2, up2])
        deconv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge2)
        deconv2 = residual_block(deconv2, 128)
        
        up1 = UpSampling2D((2, 2))(deconv2)
        up1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
        merge1 = Concatenate()([conv1, up1])
        deconv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge1)
        deconv1 = residual_block(deconv1, 64)
        
        # Output
        output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(deconv1)
        
        # Model
        model = Model(inputs=inputs, outputs=output)
        
        # Compile with the combined loss
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=combined_loss,
            metrics=['mse']
        )
        
        # Force model build by calling it with a sample input
        sample_input = np.zeros((1, 64, 64, 3), dtype=np.float32)
        model(sample_input)
        
        return model


# Function to load the pre-trained model and fine-tune it
def load_and_fine_tune_model(model_path=None):
    """
    Load pre-trained model if available, otherwise create a new one,
    and prepare it for fine-tuning
    """
    if model_path and os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        with strategy.scope():
            model = load_model(model_path, compile=False)
            
            # Recompile with new loss function
            model.compile(
                optimizer=Adam(learning_rate=1e-4),
                loss=combined_loss,
                metrics=['mse']
            )
            
            print("Model loaded successfully")
            return model
    else:
        print("Creating new model with improved architecture")
        return build_improved_model(input_shape=(64, 64, 3))
# Visualization functions
def visualize_results(model, test_patches, num_samples=4, save_path=None):
    """
    Visualize denoising results on test patches
    
    Args:
        model: Trained denoising model
        test_patches: List of clean test patches
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    # Select random samples
    indices = np.random.choice(len(test_patches), num_samples, replace=False)
    samples = [test_patches[i] for i in indices]
    
    noise_types = [
        ['gaussian'],
        ['salt_pepper'],
        ['speckle'],
        ['gaussian', 'salt_pepper']
    ]
    
    plt.figure(figsize=(15, 4 * len(noise_types)))
    
    for i, clean in enumerate(samples):
        # Apply different noise types
        for j, noise_type in enumerate(noise_types):
            noisy = apply_noise(clean, noise_type)
            
            # Denoise
            denoised = model.predict(np.expand_dims(noisy, axis=0))[0]
            
            # Calculate metrics
            psnr_val = psnr(clean, denoised, data_range=1.0)
            ssim_val = ssim(clean, denoised, data_range=1.0, channel_axis=2)
            
            # Plot
            plt.subplot(len(noise_types), 3 * num_samples, j * 3 * num_samples + i * 3 + 1)
            plt.imshow(clean)
            if i == 0:
                plt.title(f"Clean ({noise_type[0]})")
            plt.axis('off')
            
            plt.subplot(len(noise_types), 3 * num_samples, j * 3 * num_samples + i * 3 + 2)
            plt.imshow(noisy)
            if i == 0:
                plt.title("Noisy")
            plt.axis('off')
            
            plt.subplot(len(noise_types), 3 * num_samples, j * 3 * num_samples + i * 3 + 3)
            plt.imshow(denoised)
            if i == 0:
                plt.title("Denoised")
            plt.text(5, 20, f"PSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.4f}", color='white', 
                     bbox=dict(facecolor='black', alpha=0.7))
            plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()

# Function to evaluate model on test set
def evaluate_model(model, test_patches, noise_types=None):
    """
    Evaluate model performance on test patches with different noise types
    
    Args:
        model: Trained denoising model
        test_patches: List of clean test patches
        noise_types: List of noise types to evaluate
        
    Returns:
        Dictionary of evaluation results
    """
    if noise_types is None:
        noise_types = [
            ['gaussian', {'gaussian_std': 0.05}],
            ['gaussian', {'gaussian_std': 0.15}],
            ['salt_pepper', {'salt_pepper_amount': 0.05}],
            ['speckle', {'speckle_var': 0.1}]
        ]
    
    results = {}
    
    for noise_type, params in noise_types:
        psnr_values = []
        ssim_values = []
        psnr_improvements = []
        ssim_improvements = []
        
        print(f"Evaluating on {noise_type} noise...")
        for i, clean in enumerate(tqdm(test_patches[:100])):  # Use first 100 patches for evaluation
            # Apply noise
            noisy = apply_noise(clean, [noise_type], params)
            
            # Denoise
            denoised = model.predict(np.expand_dims(noisy, axis=0))[0]
            
            # Calculate metrics
            clean_psnr = psnr(clean, denoised, data_range=1.0)
            clean_ssim = ssim(clean, denoised, data_range=1.0, channel_axis=2)
            
            noisy_psnr = psnr(clean, noisy, data_range=1.0)
            noisy_ssim = ssim(clean, noisy, data_range=1.0, channel_axis=2)
            
            psnr_values.append(clean_psnr)
            ssim_values.append(clean_ssim)
            psnr_improvements.append(clean_psnr - noisy_psnr)
            ssim_improvements.append(clean_ssim - noisy_ssim)
        
        # Calculate average metrics
        results[noise_type] = {
            'psnr': np.mean(psnr_values),
            'ssim': np.mean(ssim_values),
            'psnr_improvement': np.mean(psnr_improvements),
            'ssim_improvement': np.mean(ssim_improvements)
        }
        
        print(f"  PSNR: {results[noise_type]['psnr']:.2f} dB")
        print(f"  SSIM: {results[noise_type]['ssim']:.4f}")
        print(f"  PSNR Improvement: {results[noise_type]['psnr_improvement']:.2f} dB")
        print(f"  SSIM Improvement: {results[noise_type]['ssim_improvement']:.4f}")
    
    return results

# Main function
def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    # Try to find datasets in Kaggle's input directory
    kaggle_datasets = find_kaggle_datasets() if IN_KAGGLE else None
    
    # If no datasets found in Kaggle, download them
    if not kaggle_datasets:
        download_datasets()
    
    # Get all image files for training
    image_files = get_image_files(kaggle_datasets)
    
    if len(image_files) == 0:
        print("No image files found. Please check the dataset paths.")
        return
    
    # Prepare image patches
    
    
    # Look for the existing model in Kaggle input directory
    model_paths = list(Path(INPUT_DIR).glob('**/*.h5')) + list(Path(INPUT_DIR).glob('**/*.keras'))
    model_path = model_paths[0] if model_paths else None
    
    # Load or create model
    model = load_and_fine_tune_model()
    model.summary()
    
    # Create data generators
    batch_size = 32
    train_gen = DenoiseDataGenerator(
        train_patches, 
        batch_size=32
    )
    val_gen = DenoiseDataGenerator(
        val_patches, 
        batch_size=32, 
        is_training=False,
        noise_types=['gaussian', 'salt_pepper'],
        noise_params={'gaussian_std': 0.1, 'salt_pepper_amount': 0.05}
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            str(MODEL_DIR / 'denoising_model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.keras'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        ),
        ModelCheckpoint(
            str(MODEL_DIR / 'denoising_model_best.keras'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        ),
        TensorBoard(
            log_dir=str(LOG_DIR),
            histogram_freq=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Training configuration
    epochs = 100  # Adjust based on convergence
    
    # Train the model with multi-GPU support
    with strategy.scope():
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
        )
    
    # Evaluate the model on test set
    test_results = evaluate_model(model, test_patches)
    
    # Visualize results
    save_path = WORKING_DIR / 'denoising_results_visualization.png'
    visualize_results(model, test_patches, save_path=save_path)
    
    # Save the final fine-tuned model in both .keras and .h5 formats
    final_model_keras_path = WORKING_DIR / 'fine_tuned_denoising_model.keras'
    final_model_h5_path = WORKING_DIR / 'fine_tuned_denoising_model.h5'
    
    # Save in Keras format
    model.save(final_model_keras_path)
    print(f"Fine-tuned model saved to {final_model_keras_path}")
    
    # Save in HDF5 format
    model.save(final_model_h5_path)
    print(f"Fine-tuned model saved to {final_model_h5_path}")
    
    # Optional: Save evaluation results
    import json
    results_path = WORKING_DIR / 'model_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    print(f"Evaluation results saved to {results_path}")

if __name__ == '__main__':
    main()