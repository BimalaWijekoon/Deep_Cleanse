import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt

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

class ImageDenoiserApp:
    def __init__(self, master):
        self.master = master
        master.title("Advanced Image Denoiser")
        master.geometry("800x750")

        # Load the latest model
        self.load_latest_model()

        # Create UI components
        self.create_ui()

    def load_latest_model(self):
        """
        Load the latest fine-tuned model
        """
        MODEL_DIR = 'FineTunedModelsv2'
        
        # Find model files
        model_paths = (
            [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) 
             if f.endswith('.keras') or f.endswith('.h5')]
        )
        
        if not model_paths:
            messagebox.showerror("Error", "No model found in the directory.")
            sys.exit(1)
        
        # Use the most recently modified model
        model_path = max(model_paths, key=os.path.getmtime)
        print(f"Loading model: {model_path}")
        
        # Load the model with custom loss function
        custom_objects = {
            'combined_loss': combined_loss
        }
        
        self.model = load_model(model_path, custom_objects=custom_objects)

    def create_ui(self):
        """
        Create the user interface
        """
        # Image Upload Section
        upload_frame = tk.LabelFrame(self.master, text="Upload Image")
        upload_frame.pack(padx=10, pady=10, fill='x')

        self.upload_button = tk.Button(upload_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(side='left', padx=10, pady=10)

        self.image_path_label = tk.Label(upload_frame, text="No image selected")
        self.image_path_label.pack(side='left', padx=10)

        # Noise Configuration Section
        noise_frame = tk.LabelFrame(self.master, text="Noise Configuration")
        noise_frame.pack(padx=10, pady=10, fill='x')

        # Noise Type Selection
        tk.Label(noise_frame, text="Noise Type:").grid(row=0, column=0, padx=5, pady=5)
        self.noise_type_var = tk.StringVar(value="gaussian")
        noise_types = ["gaussian", "salt_pepper", "speckle"]
        self.noise_type_dropdown = ttk.Combobox(noise_frame, textvariable=self.noise_type_var, values=noise_types)
        self.noise_type_dropdown.grid(row=0, column=1, padx=5, pady=5)

        # Noise Level Slider
        tk.Label(noise_frame, text="Noise Level:").grid(row=1, column=0, padx=5, pady=5)
        self.noise_level_var = tk.DoubleVar(value=0.3)
        self.noise_level_slider = tk.Scale(noise_frame, from_=0.01, to=0.7, resolution=0.01, 
                                            orient='horizontal', variable=self.noise_level_var,
                                            length=300)
        self.noise_level_slider.grid(row=1, column=1, padx=5, pady=5)

        # Progress Bar Section
        self.progress_frame = tk.LabelFrame(self.master, text="Processing")
        self.progress_frame.pack(padx=10, pady=10, fill='x')

        self.progress_bar = ttk.Progressbar(self.progress_frame, orient='horizontal', 
                                             length=700, mode='determinate')
        self.progress_bar.pack(padx=10, pady=10, fill='x')
        self.progress_label = tk.Label(self.progress_frame, text="")
        self.progress_label.pack(padx=10)

        # Image Preview Section
        preview_frame = tk.LabelFrame(self.master, text="Image Previews")
        preview_frame.pack(padx=10, pady=10, expand=True, fill='both')

        # Original Image Preview
        tk.Label(preview_frame, text="Original Image").grid(row=0, column=0)
        self.original_image_label = tk.Label(preview_frame)
        self.original_image_label.grid(row=1, column=0, padx=10, pady=10)

        # Noisy Image Preview
        tk.Label(preview_frame, text="Noisy Image").grid(row=0, column=1)
        self.noisy_image_label = tk.Label(preview_frame)
        self.noisy_image_label.grid(row=1, column=1, padx=10, pady=10)

        # Denoised Image Preview
        tk.Label(preview_frame, text="Denoised Image").grid(row=0, column=2)
        self.denoised_image_label = tk.Label(preview_frame)
        self.denoised_image_label.grid(row=1, column=2, padx=10, pady=10)

        # Generate Button
        self.generate_button = tk.Button(self.master, text="Generate Denoised Image", 
                                          command=self.generate_denoised_image, state=tk.DISABLED)
        self.generate_button.pack(pady=10)

    def upload_image(self):
        """
        Upload an image and display it in the original image preview
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if file_path:
            self.image_path = file_path
            self.image_path_label.config(text=os.path.basename(file_path))
            
            # Load and display original image
            original_image = Image.open(file_path)
            original_image.thumbnail((300, 300))  # Resize for preview
            self.original_photo = ImageTk.PhotoImage(original_image)
            self.original_image_label.config(image=self.original_photo)
            
            # Enable generate button
            self.generate_button.config(state=tk.NORMAL)

    def preprocess_image(self, image):
        """
        Preprocess image for model input
        """
        # Normalize to [0, 1]
        image_array = np.array(image) / 255.0

        # Ensure image is tiled into 64x64 patches
        h, w = image_array.shape[:2]
        
        # Determine the size of the padded image
        pad_h = ((h + 63) // 64) * 64
        pad_w = ((w + 63) // 64) * 64
        
        # Create a padded image
        padded_image = np.zeros((pad_h, pad_w, 3), dtype=image_array.dtype)
        padded_image[:h, :w] = image_array
        
        return padded_image, (h, w)

    def advanced_noise_application(self, image):
        """
        Advanced noise application with more realistic noise generation
        """
        noise_type = self.noise_type_var.get()
        noise_level = self.noise_level_var.get()
        
        # Make a copy to avoid modifying the original
        noisy_image = image.copy()
        
        # Get image shape
        h, w, c = noisy_image.shape
        
        if noise_type == 'gaussian':
            # More sophisticated gaussian noise with spatial variation
            noise = np.random.normal(0, noise_level, (h, w, c))
            
            # Create a spatial variation mask
            spatial_mask = np.random.uniform(0.5, 1.5, (h, w, 1))
            noise *= spatial_mask
            
            noisy_image = noisy_image + noise
        
        elif noise_type == 'salt_pepper':
            # Salt and pepper noise implementation
            s_vs_p = 0.5  # Ratio of salt vs. pepper noise
            amount = noise_level  # Overall density of noise
            
            # Salt (white) noise
            num_salt = np.ceil(amount * image.size * s_vs_p)
            # Generate coordinates for salt noise
            salt_coords = (np.random.randint(0, h, int(num_salt)),
                           np.random.randint(0, w, int(num_salt)),
                           np.random.randint(0, c, int(num_salt)))
            noisy_image[salt_coords] = 1
            
            # Pepper (black) noise
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            # Generate coordinates for pepper noise
            pepper_coords = (np.random.randint(0, h, int(num_pepper)),
                             np.random.randint(0, w, int(num_pepper)),
                             np.random.randint(0, c, int(num_pepper)))
            noisy_image[pepper_coords] = 0
        
        elif noise_type == 'speckle':
            # More advanced speckle noise with multiplicative variation
            noise = np.random.normal(1, noise_level, noisy_image.shape)
            noisy_image *= noise
        
        # Clip values to valid range [0,1]
        noisy_image = np.clip(noisy_image, 0.0, 1.0)
        
        return noisy_image

    def generate_denoised_image(self):
        """
        Generate denoised image using the loaded model with progress tracking
        """
        try:
            # Reset progress bar
            self.progress_bar['value'] = 0
            self.progress_label.config(text="Starting image processing...")
            self.master.update_idletasks()

            # Load the original image
            original_image = Image.open(self.image_path)
            
            # Preprocess the image
            preprocessed_image, original_size = self.preprocess_image(original_image)
            
            # Update progress
            self.progress_bar['value'] = 10
            self.progress_label.config(text="Image preprocessed...")
            self.master.update_idletasks()

            # Apply advanced noise
            noisy_image = self.advanced_noise_application(preprocessed_image)
            
            # Display noisy image
            noisy_display = Image.fromarray((noisy_image * 255).astype(np.uint8))
            noisy_display.thumbnail((300, 300))
            self.noisy_photo = ImageTk.PhotoImage(noisy_display)
            self.noisy_image_label.config(image=self.noisy_photo)
            
            # Update progress
            self.progress_bar['value'] = 30
            self.progress_label.config(text="Noise applied...")
            self.master.update_idletasks()

            # Process image in patches using the model
            h, w, _ = preprocessed_image.shape
            denoised_image = np.zeros_like(preprocessed_image)
            
            # Calculate total number of patches
            total_patches = (h // 64) * (w // 64)
            processed_patches = 0
            
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
                        denoised_patch = self.model.predict(np.expand_dims(patch, axis=0))[0]
                    
                    # Place denoised patch back into the image
                        denoised_image[y:y+64, x:x+64] = denoised_patch
                    
                    # Update progress
                        processed_patches += 1
                        progress = 30 + min(70, (processed_patches / total_patches) * 70)
                    self.progress_bar['value'] = progress
                    self.progress_label.config(text=f"Processing patches: {processed_patches}/{total_patches}")
                    self.master.update_idletasks()
            
            # Crop back to original size
            denoised_image = denoised_image[:original_size[0], :original_size[1]]
            
            # Display denoised image
            denoised_display = Image.fromarray((denoised_image * 255).astype(np.uint8))
            denoised_display.thumbnail((300, 300))
            self.denoised_photo = ImageTk.PhotoImage(denoised_display)
            self.denoised_image_label.config(image=self.denoised_photo)
            
            # Final progress update
            self.progress_bar['value'] = 100
            self.progress_label.config(text="Image denoising complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.progress_bar['value'] = 0
            self.progress_label.config(text="Error occurred during processing")

def main():
    root = tk.Tk()
    app = ImageDenoiserApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()