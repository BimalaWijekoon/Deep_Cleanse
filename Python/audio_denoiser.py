import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import soundfile as sf
import io
import base64
import uuid
from datetime import datetime
import tempfile

# Custom loss function for audio model
def combined_audio_loss(y_true, y_pred):
    """
    Combined loss function used during model training
    Combines MSE and SNR losses
    """
    # MSE Loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Signal-to-noise ratio loss
    def snr_loss(y_true, y_pred):
        noise = y_true - y_pred
        signal_power = tf.reduce_sum(tf.square(y_true), axis=-1)
        noise_power = tf.reduce_sum(tf.square(noise), axis=-1)
        snr = 10 * tf.math.log(signal_power / (noise_power + 1e-10)) / tf.math.log(10.0)
        return -tf.reduce_mean(snr)  # Negative because we want to maximize SNR
    
    snr_l = snr_loss(y_true, y_pred)
    
    # Combined loss (you can adjust the weights)
    return 0.7 * mse_loss + 0.3 * snr_l

class AudioAnalyzer:
    """
    Class to analyze and detect noise in audio files
    """
    def __init__(self):
        pass
    
    def analyze_noise(self, audio_data, sample_rate):
        """
        Analyze audio for noise levels and types
        
        Args:
            audio_data: Audio signal array
            sample_rate: Sampling rate of the audio
            
        Returns:
            dict: Dictionary with noise analysis results
        """
        results = {
            'has_noise': False,
            'noise_types': [],
            'noise_levels': {},
            'overall_noise_level': 0.0,
            'recommendations': '',
            'duration': len(audio_data) / sample_rate
        }
        
        # Calculate signal-to-noise ratio (SNR)
        def estimate_snr(audio):
            # Estimate noise from silent parts or high-frequency components
            # This is a simple implementation that could be improved
            
            # Method 1: Use the quietest 10% of the signal as noise estimate
            audio_sorted = np.sort(np.abs(audio))
            noise_threshold = audio_sorted[int(len(audio_sorted) * 0.1)]
            noise_mask = np.abs(audio) <= noise_threshold
            
            if np.sum(noise_mask) > 0:
                noise_estimate = audio[noise_mask]
                signal_power = np.mean(np.square(audio))
                noise_power = np.mean(np.square(noise_estimate))
                if noise_power > 0:
                    snr = 10 * np.log10(signal_power / noise_power)
                    return snr
            
            # Fallback: use spectral subtraction method
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Assume the lowest 5% of frequency bins are noise
            sorted_bins = np.sort(np.mean(magnitude, axis=1))
            noise_threshold = sorted_bins[int(len(sorted_bins) * 0.05)]
            
            noise_mask = np.mean(magnitude, axis=1) <= noise_threshold
            noise_profile = magnitude[noise_mask]
            
            if len(noise_profile) > 0:
                signal_power = np.mean(np.square(magnitude))
                noise_power = np.mean(np.square(noise_profile))
                if noise_power > 0:
                    snr = 10 * np.log10(signal_power / noise_power)
                    return snr
            
            # If all else fails, return a default value
            return 15.0  # Moderate SNR
        
        # Estimate SNR
        estimated_snr = estimate_snr(audio_data)
        
        # Calculate spectral features for noise classification
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)[0]
        
        # Determine noise types and levels
        # White noise: high flatness, high bandwidth
        # Hum/buzz: low bandwidth, distinct peaks at low frequencies
        # Background noise: moderate features across the board
        
        avg_flatness = np.mean(spectral_flatness)
        avg_bandwidth = np.mean(spectral_bandwidth)
        
        # Check for white noise
        if avg_flatness > 0.2:  # High spectral flatness indicates white noise
            results['has_noise'] = True
            results['noise_types'].append('white_noise')
            white_noise_level = min(1.0, avg_flatness * 5) * 100  # Scale to percentage
            results['noise_levels']['white_noise'] = white_noise_level
        
        # Check for hum/buzz (usually power line interference at 50/60 Hz)
        # This requires more sophisticated analysis, but we'll use a simple approach
        stft = librosa.stft(audio_data)
        magnitude = np.abs(stft)
        freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
        
        # Look for peaks in low frequencies (below 300 Hz)
        low_freq_mask = freq_bins < 300
        low_freq_magnitudes = np.mean(magnitude[low_freq_mask], axis=1)
        high_freq_magnitudes = np.mean(magnitude[~low_freq_mask], axis=1)
        
        low_high_ratio = np.mean(low_freq_magnitudes) / (np.mean(high_freq_magnitudes) + 1e-10)
        
        if low_high_ratio > 2.0:  # Significantly more energy in low frequencies
            results['has_noise'] = True
            results['noise_types'].append('hum_buzz')
            hum_level = min(1.0, (low_high_ratio - 2.0) / 8.0) * 100  # Scale to percentage
            results['noise_levels']['hum_buzz'] = hum_level
        
        # Check for background noise (based on SNR)
        if estimated_snr < 20:  # Below 20dB SNR indicates notable background noise
            results['has_noise'] = True
            results['noise_types'].append('background_noise')
            bg_noise_level = min(1.0, (20 - estimated_snr) / 20) * 100  # Scale to percentage
            results['noise_levels']['background_noise'] = bg_noise_level
        
        # Calculate overall noise level
        if results['noise_levels']:
            results['overall_noise_level'] = sum(results['noise_levels'].values()) / len(results['noise_levels'])
            
            # Prepare recommendations
            if results['overall_noise_level'] > 50:
                results['recommendations'] = "High noise levels detected. Multiple denoising rounds recommended."
            elif results['overall_noise_level'] > 20:
                results['recommendations'] = "Moderate noise detected. One round of denoising should be sufficient."
            else:
                results['recommendations'] = "Low noise levels detected. Light denoising recommended."
        else:
            results['recommendations'] = "No significant noise detected. Audio appears to be clear."
        
        # Additional metrics
        results['snr'] = float(estimated_snr)
        results['spectral_flatness'] = float(avg_flatness)
        
        return results

class AudioDenoiser:
    def __init__(self):
        self.model = None
        self.audio_analyzer = AudioAnalyzer()
        self.load_model()
        # Store processed audio in memory
        self.audio_cache = {}
    
    def load_model(self):
        """
        Load the fine-tuned model for audio denoising
        """
        # Current file directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(current_dir, 'AuidoDenoise', 'FineTunedModelsv2')
        
        try:
            # Find model files
            if not os.path.exists(MODEL_DIR):
                print(f"Error: Model directory {MODEL_DIR} not found.")
                print("Using dummy model for audio denoising.")
                self.model = self._create_dummy_model()
                return
                
            model_paths = [
                os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) 
                if f.endswith('.keras') or f.endswith('.h5')
            ]
            
            if not model_paths:
                print("Error: No model files found in AudioDenoise directory.")
                print("Using dummy model for audio denoising.")
                self.model = self._create_dummy_model()
                return
            
            # Use the most recently modified model
            model_path = max(model_paths, key=os.path.getmtime)
            print(f"Loading audio model: {model_path}")
            
            # Load the model with custom loss function
            custom_objects = {
                'combined_audio_loss': combined_audio_loss,
                'combined_loss': combined_audio_loss  # Add this for compatibility with image models
            }
            
            self.model = load_model(model_path, custom_objects=custom_objects)
            print("Audio model loaded successfully")
            
        except Exception as e:
            print(f"Error loading audio model: {str(e)}")
            print("Using dummy model for audio denoising.")
            self.model = self._create_dummy_model()
    
    def _create_dummy_model(self):
        """
        Create a simple dummy model for audio denoising when no model is available
        """
        print("Creating dummy audio denoising model")
        
        # Define a simple model that just returns the input (identity function)
        class DummyModel:
            def predict(self, input_data, verbose=0):
                # Simply return the input slightly modified to simulate denoising
                # Apply some basic noise reduction by smoothing the signal
                result = input_data.copy()
                if len(result.shape) >= 2 and result.shape[1] > 10:
                    # Apply a simple moving average filter
                    window_size = 5
                    for i in range(result.shape[0]):
                        temp = np.convolve(result[i].flatten(), 
                                          np.ones(window_size)/window_size, 
                                          mode='same')
                        result[i] = temp.reshape(result[i].shape)
                return result
        
        return DummyModel()
    
    def preprocess_audio(self, audio_data, sample_rate):
        """
        Preprocess audio for model input
        """
        # Normalize audio
        normalized_audio = audio_data / np.max(np.abs(audio_data))
        
        # Ensure consistent sample rate (the model expects 16000 Hz)
        target_sr = 16000  # Model expects 16kHz
        if sample_rate != target_sr:
            normalized_audio = librosa.resample(normalized_audio, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr
        
        # Split into chunks for processing
        chunk_size = 16000  # Model expects chunks of 16000 samples
        audio_chunks = []
        
        # For very short audio, pad to 16000
        if len(normalized_audio) < chunk_size:
            padded_audio = np.pad(normalized_audio, (0, chunk_size - len(normalized_audio)))
            model_input = padded_audio.reshape(1, chunk_size, 1)
            audio_chunks.append(model_input)
        else:
            # Process longer audio in chunks
            # Use non-overlapping chunks for simplicity
            for i in range(0, len(normalized_audio), chunk_size):
                chunk = normalized_audio[i:i+chunk_size]
                # Pad the last chunk if needed
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                # Reshape for model input (batch_size, time_steps, features)
                model_input = chunk.reshape(1, chunk_size, 1)
                audio_chunks.append(model_input)
        
        return audio_chunks, sample_rate, len(normalized_audio)
    
    def analyze_audio(self, audio_file):
        """
        Analyze an uploaded audio file for noise
        
        Args:
            audio_file: Uploaded audio file
            
        Returns:
            dict: Noise analysis results
        """
        try:
            # Save the file temporarily to load it with librosa
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                # Use a try-except block to handle different file object types
                try:
                    # For flask.FileStorage objects
                    audio_file.save(temp_file.name)
                except AttributeError:
                    # For regular file objects or if save method isn't available
                    if hasattr(audio_file, 'read'):
                        # If it's a file-like object with read method
                        content = audio_file.read()
                        temp_file.write(content)
                    else:
                        # If it's a file path string
                        with open(audio_file, 'rb') as f:
                            temp_file.write(f.read())
                
                temp_filename = temp_file.name
            
            # Load audio file
            try:
                audio_data, sample_rate = librosa.load(temp_filename, sr=None)
            except Exception as e:
                # Provide more detailed error information
                print(f"Error loading audio file with librosa: {str(e)}")
                return {'error': f"Failed to process audio file: {str(e)}. Please ensure it's a valid audio format."}
            
            # Remove the temporary file
            try:
                os.unlink(temp_filename)
            except Exception as e:
                print(f"Warning: Failed to delete temporary file {temp_filename}: {str(e)}")
            
            # Generate a unique ID for this audio
            audio_id = str(uuid.uuid4())
            
            # Get filename safely
            if hasattr(audio_file, 'filename'):
                filename = audio_file.filename
            else:
                # Use a default name if filename attribute isn't available
                filename = f"uploaded_audio_{audio_id}.wav"
            
            # Store the audio for later use
            self.audio_cache[audio_id] = {
                'original': {
                    'data': audio_data,
                    'sample_rate': sample_rate
                },
                'current': {
                    'data': audio_data,
                    'sample_rate': sample_rate
                },
                'denoising_round': 0,
                'file_name': filename,
                'history': []  # Initialize empty history array
            }
            
            # Analyze noise
            analysis_results = self.audio_analyzer.analyze_noise(audio_data, sample_rate)
            
            # Add audio ID and metadata to results
            analysis_results['audio_id'] = audio_id
            analysis_results['file_name'] = filename
            analysis_results['sample_rate'] = int(sample_rate)
            analysis_results['channels'] = 1  # Librosa loads as mono
            
            # Generate a preview of the audio as base64 for the web player
            preview_data = self.generate_audio_preview(audio_data, sample_rate)
            analysis_results['audio_preview'] = preview_data
            
            return analysis_results
            
        except Exception as e:
            # Add detailed error logging
            print(f"Error in analyze_audio: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def denoise_audio(self, audio_id):
        """
        Denoise the audio using the loaded model
        
        Args:
            audio_id: ID of the audio to denoise
            
        Returns:
            dict: Denoising results including the denoised audio preview
        """
        try:
            # Debug info for cache contents
            print(f"Attempting to denoise audio with ID: {audio_id}")
            print(f"Audio cache keys: {list(self.audio_cache.keys())}")
            
            # Check if audio exists in cache
            if audio_id not in self.audio_cache:
                print(f"Error: Audio ID '{audio_id}' not found in cache")
                return {'error': f"Audio not found. Please analyze the audio first. ID: {audio_id}"}
            
            # Get audio data
            audio_data = self.audio_cache[audio_id]
            print(f"Retrieved audio data with keys: {list(audio_data.keys())}")
            
            current_audio = audio_data['current']['data']
            sample_rate = audio_data['current']['sample_rate']
            
            print(f"Audio shape: {current_audio.shape}, Sample rate: {sample_rate}, Duration: {len(current_audio)/sample_rate:.2f}s")
            
            # Increment denoising round
            audio_data['denoising_round'] += 1
            current_round = audio_data['denoising_round']
            
            # Preprocess audio
            audio_chunks, processed_sr, original_length = self.preprocess_audio(current_audio, sample_rate)
            print(f"Preprocessed into {len(audio_chunks)} chunks, original length: {original_length}")
            
            # Process each chunk
            denoised_chunks = []
            for i, chunk in enumerate(audio_chunks):
                try:
                    print(f"Processing chunk {i+1}/{len(audio_chunks)}, shape: {chunk.shape}")
                    denoised_chunk = self.model.predict(chunk, verbose=0)
                    # Extract the denoised audio, removing batch and channel dimensions
                    denoised_audio_chunk = denoised_chunk[0, :, 0]
                    denoised_chunks.append(denoised_audio_chunk)
                    print(f"Processed chunk {i+1}, output shape: {denoised_audio_chunk.shape}")
                except Exception as e:
                    print(f"Error in model prediction for chunk {i+1}: {str(e)}")
                    return {'error': f"Model prediction failed on chunk {i+1}: {str(e)}"}
            
            # Concatenate all chunks
            denoised_audio = np.concatenate(denoised_chunks)
            # Trim back to original length to avoid padding artifacts
            denoised_audio = denoised_audio[:original_length]
            
            print(f"Final denoised audio shape: {denoised_audio.shape}, duration: {len(denoised_audio)/processed_sr:.2f}s")
            
            # Normalize output
            denoised_audio = denoised_audio / np.max(np.abs(denoised_audio))
            
            # Calculate improvement metrics
            try:
                metrics = self.calculate_improvement_metrics(
                    audio_data['original']['data'], 
                    denoised_audio
                )
                print("Successfully calculated improvement metrics")
            except Exception as e:
                print(f"Error calculating metrics: {str(e)}")
                metrics = {}
            
            # Generate a preview of the denoised audio
            try:
                denoised_preview = self.generate_audio_preview(denoised_audio, processed_sr)
                print("Successfully generated audio preview")
                
                # Generate waveform data for visualization
                original_waveform = self.generate_waveform_data(audio_data['original']['data'])
                denoised_waveform = self.generate_waveform_data(denoised_audio)
                print("Successfully generated waveform data")
            except Exception as e:
                print(f"Error generating preview or waveform: {str(e)}")
                denoised_preview = None
                original_waveform = []
                denoised_waveform = []
            
            # Store the current state in history
            if 'history' not in audio_data:
                audio_data['history'] = []
                
            # Only save previous round to history before updating current
            if current_round > 1:  # We only need to save from round 2 onwards
                previous_round_data = {
                    'data': np.copy(current_audio),
                    'sample_rate': sample_rate,
                    'round': current_round - 1
                }
                
                # Add to history - position is round-1 since array is 0-indexed
                if len(audio_data['history']) < current_round - 1:
                    # First time adding this round
                    audio_data['history'].append(previous_round_data)
                else:
                    # Update existing entry
                    audio_data['history'][current_round - 2] = previous_round_data
                    
            # Update the current state for potential next rounds
            audio_data['current']['data'] = denoised_audio
            audio_data['current']['sample_rate'] = processed_sr
            
            # Prepare response
            response = {
                'audio_id': audio_id,
                'denoising_round': current_round,
                'audio_preview': denoised_preview,
                'sample_rate': int(processed_sr),
                'metrics': metrics,
                'duration': float(len(denoised_audio) / processed_sr),
                'waveform_data': {
                    'original': original_waveform,
                    'denoised': denoised_waveform
                }
            }
            
            print("Successfully prepared denoising response")
            return response
            
        except Exception as e:
            print(f"Unexpected error in denoise_audio: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def calculate_improvement_metrics(self, original, denoised):
        """
        Calculate and return audio improvement metrics
        """
        try:
            # Ensure same length for comparison
            min_length = min(len(original), len(denoised))
            original = original[:min_length]
            denoised = denoised[:min_length]
            
            # Calculate metrics
            # Signal-to-Noise Ratio (SNR) improvement
            noise_original = np.mean(np.square(original))
            noise_removed = np.mean(np.square(original - denoised))
            snr_improvement = 10 * np.log10((noise_original + 1e-10) / (noise_removed + 1e-10))
            
            # Spectral distance reduction
            orig_stft = np.abs(librosa.stft(original))
            denoised_stft = np.abs(librosa.stft(denoised))
            spectral_distance = np.mean(np.abs(orig_stft - denoised_stft))
            
            # Energy preservation (should be close to 1)
            energy_ratio = np.sum(np.square(denoised)) / (np.sum(np.square(original)) + 1e-10)
            
            # Calculate PESQ-like metric (simplified)
            # True PESQ requires specialized libraries
            energy_original = np.sum(np.square(original))
            energy_error = np.sum(np.square(original - denoised))
            pesq_like = -10 * np.log10(energy_error / (energy_original + 1e-10))
            
            # Check if the improvement is significant
            low_improvement = snr_improvement < 3.0  # Less than 3dB improvement is not very noticeable
            
            return {
                'snr_improvement': float(snr_improvement),
                'spectral_distance': float(spectral_distance),
                'energy_ratio': float(energy_ratio),
                'quality_score': float(pesq_like),
                'low_improvement': bool(low_improvement)
            }
            
        except Exception as e:
            print(f"Error calculating audio metrics: {e}")
            return {}
    
    def generate_audio_preview(self, audio_data, sample_rate):
        """
        Generate a base64-encoded audio preview for web playback
        """
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                # Save audio to the temp file
                sf.write(temp_file.name, audio_data, sample_rate, format='WAV')
                
                # Read the file as binary data
                temp_file.seek(0)
                with open(temp_file.name, 'rb') as audio_file:
                    audio_binary = audio_file.read()
                
                # Convert to base64
                audio_base64 = base64.b64encode(audio_binary).decode('utf-8')
                
                # Add data URI prefix
                audio_preview = f"data:audio/wav;base64,{audio_base64}"
            
            # Remove the temporary file
            os.unlink(temp_file.name)
            
            return audio_preview
            
        except Exception as e:
            print(f"Error generating audio preview: {e}")
            return None
    
    def get_audio_preview(self, audio_id):
        """
        Get a preview of the current state of the audio
        """
        try:
            # Check if audio exists in cache
            if audio_id not in self.audio_cache:
                return {'error': 'Audio not found'}
            
            # Get audio data
            audio_data = self.audio_cache[audio_id]
            current_audio = audio_data['current']['data']
            sample_rate = audio_data['current']['sample_rate']
            
            # Generate preview
            preview_data = self.generate_audio_preview(current_audio, sample_rate)
            
            return {
                'audio_id': audio_id,
                'audio_preview': preview_data,
                'denoising_round': audio_data['denoising_round']
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def save_audio(self, audio_id, filename=None, format_type='wav', version='current'):
        """
        Get the denoised audio for saving
        
        Args:
            audio_id: ID of the audio to save
            filename: Optional filename to use (without extension)
            format_type: Output format ('wav' or 'mp3')
            version: Which version to save ('original', 'current', or round number)
            
        Returns:
            Response with audio data for download
        """
        try:
            print(f"Save audio request: audio_id={audio_id}, format={format_type}, version={version}")
            
            # Check if audio exists in cache
            if audio_id not in self.audio_cache:
                print(f"Error: Audio ID '{audio_id}' not found in cache")
                return {'error': f"Audio ID not found: {audio_id}"}
            
            # Get audio data
            audio_data = self.audio_cache[audio_id]
            print(f"Found audio in cache with keys: {list(audio_data.keys())}")
            
            # Select the appropriate audio version
            if version == 'original':
                # Original unprocessed audio
                current_audio = audio_data['original']['data']
                sample_rate = audio_data['original']['sample_rate']
                print(f"Saving original audio, shape: {current_audio.shape}, rate: {sample_rate}")
            elif version == 'current':
                # Latest processed version
                current_audio = audio_data['current']['data']
                sample_rate = audio_data['current']['sample_rate']
                print(f"Saving current audio (round {audio_data['denoising_round']}), shape: {current_audio.shape}, rate: {sample_rate}")
            else:
                # Try to parse as a specific round number
                try:
                    requested_round = int(version)
                    if 'history' not in audio_data or requested_round <= 0 or requested_round > len(audio_data.get('history', [])):
                        # If round not found, fall back to current
                        print(f"Requested round {requested_round} not found, using current audio")
                        current_audio = audio_data['current']['data']
                        sample_rate = audio_data['current']['sample_rate']
                    else:
                        # Get the specific round from history
                        history_item = audio_data['history'][requested_round - 1]
                        current_audio = history_item['data']
                        sample_rate = history_item['sample_rate']
                        print(f"Saving audio from round {requested_round}, shape: {current_audio.shape}, rate: {sample_rate}")
                except ValueError:
                    # If version is not a valid number, fall back to current
                    print(f"Invalid version: {version}, using current audio")
                    current_audio = audio_data['current']['data']
                    sample_rate = audio_data['current']['sample_rate']
            
            # Verify that we have valid audio data
            if current_audio is None or len(current_audio) == 0:
                print(f"Error: No audio data found for {audio_id}, version {version}")
                return {'error': f"No audio data found for specified version: {version}"}
                
            # Ensure sample rate is valid
            if sample_rate <= 0:
                print(f"Error: Invalid sample rate {sample_rate}")
                sample_rate = 44100  # Fallback to a standard rate
            
            # Generate a filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                original_name = os.path.splitext(audio_data['file_name'])[0]
                if version == 'original':
                    filename = f"{original_name}_original_{timestamp}"
                elif version == 'current':
                    filename = f"{original_name}_denoised_{timestamp}"
                else:
                    filename = f"{original_name}_denoised_round{version}_{timestamp}"
            
            print(f"Generated filename: {filename}")
            
            # Create in-memory file-like object
            audio_io = io.BytesIO()
            
            # Save in requested format
            if format_type.lower() == 'mp3':
                # MP3 export requires additional libraries
                try:
                    # Check if pydub is installed
                    import importlib
                    if importlib.util.find_spec('pydub') is None:
                        print("Error: pydub module not found, cannot export to MP3")
                        return {'error': "MP3 export requires pydub module. Please install it with 'pip install pydub'"}
                    
                    import pydub
                    from pydub import AudioSegment
                    
                    # Save as WAV first
                    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    temp_wav_path = temp_wav.name
                    temp_wav.close()  # Close the file so we can write to it
                    
                    print(f"Saving temporary WAV file: {temp_wav_path}")
                    sf.write(temp_wav_path, current_audio, sample_rate, format='WAV')
                    
                    # Convert to MP3
                    print("Converting to MP3 using pydub")
                    audio_segment = AudioSegment.from_wav(temp_wav_path)
                    audio_segment.export(audio_io, format="mp3")
                    
                    # Cleanup
                    os.unlink(temp_wav_path)
                    
                    # Make sure we're at the start of the file
                    audio_io.seek(0)
                    print(f"Successfully created MP3 file, size: {len(audio_io.getvalue())} bytes")
                    return audio_io, f"{filename}.mp3"
                    
                except ImportError as e:
                    print(f"Import error for MP3 export: {str(e)}")
                    print("Warning: pydub not installed, falling back to WAV format")
                    format_type = 'wav'
                except Exception as e:
                    print(f"Error during MP3 export: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return {'error': f"Failed to create MP3: {str(e)}"}
            
            # Default to WAV format
            if format_type.lower() == 'wav':
                try:
                    print(f"Creating WAV file with data shape: {current_audio.shape}, sample rate: {sample_rate}")
                    
                    # Normalize audio data to avoid clipping (-1.0 to 1.0 range)
                    if np.max(np.abs(current_audio)) > 0:
                        normalized_audio = current_audio / np.max(np.abs(current_audio))
                    else:
                        normalized_audio = current_audio
                    
                    # Try using soundfile first
                    try:
                        sf.write(audio_io, normalized_audio, sample_rate, format='WAV')
                    except Exception as sf_error:
                        print(f"Error with soundfile: {str(sf_error)}, trying alternate method")
                        
                        # Alternate method: create a temporary WAV file and read it back
                        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                        temp_wav_path = temp_wav.name
                        temp_wav.close()
                        
                        try:
                            # Use scipy.io.wavfile as a fallback
                            from scipy.io import wavfile
                            # Convert to int16 for WAV format
                            int_data = (normalized_audio * 32767).astype(np.int16)
                            wavfile.write(temp_wav_path, sample_rate, int_data)
                            
                            # Read the temp file into the BytesIO
                            with open(temp_wav_path, 'rb') as f:
                                audio_io.write(f.read())
                        finally:
                            # Clean up temp file
                            if os.path.exists(temp_wav_path):
                                os.unlink(temp_wav_path)
                    
                    audio_io.seek(0)
                    file_size = len(audio_io.getvalue())
                    print(f"Successfully created WAV file, size: {file_size} bytes")
                    
                    if file_size == 0:
                        return {'error': "Failed to create WAV file (empty file)"}
                        
                    return audio_io, f"{filename}.wav"
                except Exception as e:
                    print(f"Error creating WAV file: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return {'error': f"Failed to create WAV: {str(e)}"}
                
            # Fallback
            return {'error': f"Unsupported format: {format_type}"}
            
        except Exception as e:
            print(f"Error saving audio: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def generate_waveform_data(self, audio_data, num_points=500):
        """
        Generate waveform data points for visualization
        
        Args:
            audio_data: Audio signal array
            num_points: Number of data points to return for visualization
            
        Returns:
            list: List of waveform data points
        """
        try:
            # Resample to desired number of points
            if len(audio_data) > num_points:
                # Take evenly spaced samples
                indices = np.linspace(0, len(audio_data) - 1, num_points, dtype=int)
                waveform_data = audio_data[indices].tolist()
            else:
                # If audio is shorter than num_points, use all points
                waveform_data = audio_data.tolist()
            
            return waveform_data
        except Exception as e:
            print(f"Error generating waveform data: {str(e)}")
            return [] 