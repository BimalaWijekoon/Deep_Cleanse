import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import io
import base64
import uuid
from datetime import datetime

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

class NoiseAnalyzer:
    """
    Class to analyze and detect noise in images
    """
    def __init__(self):
        pass
    
    def analyze_noise(self, image):
        """
        Analyze image and detect noise types and levels
        
        Args:
            image: Normalized image array (0-1 float values)
            
        Returns:
            dict: Dictionary with noise analysis results
        """
        # Convert to grayscale for noise analysis
        if len(image.shape) == 3:
            gray_img = np.mean(image, axis=2)
        else:
            gray_img = image
            
        results = {
            'has_noise': False,
            'noise_types': [],
            'noise_levels': {},
            'overall_noise_level': 0.0,
            'recommendations': ''
        }
        
        # Check for salt and pepper noise
        salt_pixels = np.sum(gray_img > 0.95) / gray_img.size
        pepper_pixels = np.sum(gray_img < 0.05) / gray_img.size
        sp_ratio = salt_pixels + pepper_pixels
        
        if sp_ratio > 0.001:
            results['has_noise'] = True
            results['noise_types'].append('salt_pepper')
            results['noise_levels']['salt_pepper'] = min(1.0, sp_ratio * 100)
        
        # Compute local variance to detect gaussian and speckle noise
        local_var = self.compute_local_variance(gray_img)
        avg_var = np.mean(local_var)
        
        # Noise level estimation based on variance
        if avg_var > 0.0005:
            results['has_noise'] = True
            
            # Try to distinguish between gaussian and speckle
            # Analyze skewness of local variance distribution
            flattened_var = local_var.flatten()
            skewness = np.mean((flattened_var - np.mean(flattened_var))**3) / (np.std(flattened_var)**3)
            
            if skewness > 1.0:
                # More likely to be speckle noise
                results['noise_types'].append('speckle')
                results['noise_levels']['speckle'] = min(1.0, avg_var * 100)
            else:
                # More likely to be gaussian noise
                results['noise_types'].append('gaussian')
                results['noise_levels']['gaussian'] = min(1.0, avg_var * 50)
        
        # Calculate overall noise level (weighted average of detected types)
        if results['noise_levels']:
            results['overall_noise_level'] = sum(results['noise_levels'].values()) / len(results['noise_levels'])
            
            # Prepare recommendations
            if results['overall_noise_level'] > 0.5:
                results['recommendations'] = "High noise levels detected. Multiple denoising rounds recommended."
            elif results['overall_noise_level'] > 0.2:
                results['recommendations'] = "Moderate noise detected. One round of denoising should be sufficient."
            else:
                results['recommendations'] = "Low noise levels detected. Light denoising recommended."
        else:
            results['recommendations'] = "No significant noise detected. Image appears to be clear."
        
        return results
    
    def compute_local_variance(self, gray_img, window_size=5):
        """
        Compute local variance with sliding window
        """
        # Pad the image
        pad_size = window_size // 2
        padded_img = np.pad(gray_img, pad_size, mode='reflect')
        
        # Create output array
        h, w = gray_img.shape
        var_map = np.zeros((h, w))
        
        # Compute variance in local windows
        for i in range(h):
            for j in range(w):
                window = padded_img[i:i+window_size, j:j+window_size]
                var_map[i, j] = np.var(window)
                
        return var_map

class ImageDenoiser:
    def __init__(self):
        self.model = None
        self.noise_analyzer = NoiseAnalyzer()
        self.load_model()
        # Store processed images in memory
        self.image_cache = {}
    
    def load_model(self):
        """
        Load the fine-tuned model
        """
        # Current file directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(current_dir, 'ImageDenoise', 'FineTunedModelsv2')
        
        try:
            # Find model files
            model_paths = [
                os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) 
                if f.endswith('.keras') or f.endswith('.h5')
            ]
            
            if not model_paths:
                print("Error: No image model found in the directory.")
                sys.exit(1)
            
            # Use the most recently modified model
            model_path = max(model_paths, key=os.path.getmtime)
            print(f"Loading image model: {model_path}")
            
            # Load the model with custom loss function
            custom_objects = {
                'combined_loss': combined_loss
            }
            
            self.model = load_model(model_path, custom_objects=custom_objects)
            print("Image model loaded successfully")
            
        except Exception as e:
            print(f"Error loading image model: {str(e)}")
            sys.exit(1)
    
    def preprocess_image(self, image_array):
        """
        Preprocess image for model input by ensuring proper dimensions and tiling
        """
        # Ensure image is tiled into 64x64 patches
        h, w = image_array.shape[:2]
        
        # Determine the size of the padded image
        pad_h = ((h + 63) // 64) * 64
        pad_w = ((w + 63) // 64) * 64
        
        # Create a padded image
        padded_image = np.zeros((pad_h, pad_w, 3), dtype=image_array.dtype)
        padded_image[:h, :w] = image_array
        
        return padded_image, (h, w)
    
    def analyze_image(self, image_data):
        """
        Analyze the image for noise
        
        Args:
            image_data: Base64 encoded image data
            
        Returns:
            dict: Noise analysis results
        """
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            img = Image.open(io.BytesIO(image_bytes))
            
            # Generate a unique ID for this image
            image_id = str(uuid.uuid4())
            
            # Convert image to numpy array and normalize
            img_array = np.array(img) / 255.0
            
            # Resize if the image is too large
            max_dimension = 1024
            h, w = img_array.shape[:2]
            if h > max_dimension or w > max_dimension:
                scale_factor = max_dimension / max(h, w)
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                img_array = cv2.resize(img_array, (new_w, new_h))
            
            # Store the processed image for later use
            self.image_cache[image_id] = {
                'original': img,
                'processed': img_array,
                'current': img_array,  # Track the current state for multiple rounds
                'denoising_round': 0
            }
            
            # Analyze noise
            noise_results = self.noise_analyzer.analyze_noise(img_array)
            
            # Add image ID to results
            noise_results['image_id'] = image_id
            
            return noise_results
            
        except Exception as e:
            return {'error': str(e)}
    
    def denoise_image(self, image_id):
        """
        Denoise the image using the loaded model
        
        Args:
            image_id: ID of the image to denoise
            
        Returns:
            dict: Denoising results including the denoised image
        """
        try:
            # Check if image exists in cache
            if image_id not in self.image_cache:
                return {'error': 'Image not found. Please analyze the image first.'}
            
            # Get image data
            image_data = self.image_cache[image_id]
            image_array = image_data['current']
            
            # Increment denoising round
            image_data['denoising_round'] += 1
            current_round = image_data['denoising_round']
            
            # Preprocess image
            preprocessed_image, original_size = self.preprocess_image(image_array)
            
            # Process image in patches using the model
            h, w, _ = preprocessed_image.shape
            denoised_image = np.zeros_like(preprocessed_image)
            
            # Process each patch
            for y in range(0, h, 64):
                for x in range(0, w, 64):
                    # Extract 64x64 patch
                    patch = preprocessed_image[y:y+64, x:x+64]
                    
                    # Ensure patch is 64x64
                    if patch.shape[:2] != (64, 64):
                        padded_patch = np.zeros((64, 64, 3), dtype=preprocessed_image.dtype)
                        padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                        patch = padded_patch
                    
                    # Denoise patch
                    denoised_patch = self.model.predict(np.expand_dims(patch, axis=0), verbose=0)[0]
                    
                    # Place denoised patch back into the image
                    denoised_image[y:y+64, x:x+64] = denoised_patch
            
            # Crop back to original size
            denoised_image = denoised_image[:original_size[0], :original_size[1]]
            
            # Calculate improvement metrics
            metrics = self.calculate_improvement_metrics(
                np.array(image_data['original']) / 255.0, 
                denoised_image
            )
            
            # Convert denoised image to base64 for sending to client
            denoised_pil = Image.fromarray((denoised_image * 255).astype(np.uint8))
            buffered = io.BytesIO()
            denoised_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Update the current state for potential next rounds
            image_data['current'] = denoised_image
            
            # Prepare response
            response = {
                'image_id': image_id,
                'denoising_round': current_round,
                'denoised_image': f'data:image/png;base64,{img_str}',
                'metrics': metrics
            }
            
            return response
            
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_improvement_metrics(self, original, denoised):
        """
        Calculate and return improvement metrics
        """
        # Convert to grayscale if needed
        if len(original.shape) == 3:
            original_gray = np.mean(original, axis=2)
            denoised_gray = np.mean(denoised, axis=2)
        else:
            original_gray = original
            denoised_gray = denoised
        
        # Calculate metrics
        try:
            # PSNR (Peak Signal-to-Noise Ratio)
            mse = np.mean((original_gray - denoised_gray) ** 2)
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            
            # SSIM (Structural Similarity Index)
            ssim_value = tf.image.ssim(
                tf.convert_to_tensor(original, dtype=tf.float32),
                tf.convert_to_tensor(denoised, dtype=tf.float32),
                max_val=1.0
            ).numpy()
            
            # Convert TensorFlow tensor to Python scalar
            if hasattr(ssim_value, 'item'):
                ssim_value = ssim_value.item()
            else:
                ssim_value = float(ssim_value)
            
            # Variance reduction
            var_original = np.var(original_gray)
            var_denoised = np.var(denoised_gray)
            var_reduction = max(0, (var_original - var_denoised) / var_original * 100)
            
            # Convert numpy values to Python native types to ensure JSON serialization
            return {
                'psnr': float(psnr),
                'ssim': float(ssim_value),
                'var_reduction': float(var_reduction),
                'low_improvement': bool(var_reduction < 5)
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {}
    
    def save_image(self, image_id, filename=None, format='png'):
        """
        Get the denoised image for saving
        
        Args:
            image_id: ID of the image to save
            filename: Optional filename to use (without extension)
            format: Image format (png, jpg, webp)
            
        Returns:
            Response with image data for download
        """
        try:
            # Check if image exists in cache
            if image_id not in self.image_cache:
                return {'error': 'Image not found'}
            
            # Get image data
            image_data = self.image_cache[image_id]
            
            # Convert the current state to PIL Image
            denoised_pil = Image.fromarray((image_data['current'] * 255).astype(np.uint8))
            
            # Generate a filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"denoised_image_{timestamp}"
            
            # Validate and normalize format
            format = format.lower()
            if format not in ['png', 'jpg', 'jpeg', 'webp']:
                format = 'png'  # Default to PNG if invalid format
            
            # Convert jpeg to jpg for PIL
            if format == 'jpeg':
                format = 'jpg'
            
            # Prepare the image for download
            img_io = io.BytesIO()
            
            # Save with appropriate format and quality settings
            if format == 'jpg':
                # JPEG requires RGB mode
                if denoised_pil.mode == 'RGBA':
                    denoised_pil = denoised_pil.convert('RGB')
                denoised_pil.save(img_io, 'JPEG', quality=95)
            elif format == 'webp':
                denoised_pil.save(img_io, 'WEBP', quality=95)
            else:  # Default to PNG
                denoised_pil.save(img_io, 'PNG')
                
            img_io.seek(0)
            
            # Return a tuple of the BytesIO object and the filename
            return img_io, f"{filename}.{format}"
            
        except Exception as e:
            print(f"Error saving image {image_id}: {str(e)}")
            return {'error': str(e)} 