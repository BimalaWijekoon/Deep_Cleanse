import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from datetime import datetime
import io
import base64
import uuid
import json
import librosa
import soundfile as sf

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Custom loss functions for models
def combined_loss_image(y_true, y_pred):
    """Combined loss function for image denoising model"""
    # MSE Loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # SSIM Loss (1-SSIM since we want to minimize)
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    
    # Combined loss
    return 0.5 * mse_loss + 0.5 * ssim_loss

def combined_loss_audio(y_true, y_pred):
    """Combined loss function for audio denoising model"""
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
    
    # Combined loss
    return 0.7 * mse_loss + 0.3 * snr_l

# Import specific modules for image and audio processing
from image_denoiser import NoiseAnalyzer, ImageDenoiser
from audio_denoiser import AudioAnalyzer, AudioDenoiser

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Flask to use custom JSON encoder
app.json_encoder = NumpyEncoder

# Initialize denoisers
image_denoiser = ImageDenoiser()
audio_denoiser = AudioDenoiser()

# Image routes
@app.route('/api/image/analyze', methods=['POST'])
def analyze_image():
    """API endpoint to analyze an image for noise"""
    try:
        if 'image' not in request.json:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_data = request.json['image']
        results = image_denoiser.analyze_image(image_data)
        
        if 'error' in results:
            return jsonify(results), 400
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/image/denoise', methods=['POST'])
def denoise_image():
    """API endpoint to denoise an image"""
    try:
        if 'image_id' not in request.json:
            return jsonify({'error': 'No image ID provided'}), 400
        
        image_id = request.json['image_id']
        results = image_denoiser.denoise_image(image_id)
        
        if 'error' in results:
            return jsonify(results), 400
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/image/save/<image_id>', methods=['GET'])
def save_image(image_id):
    """API endpoint to save/download a denoised image"""
    try:
        filename = request.args.get('filename')
        format_type = request.args.get('format', 'png')  # Default to PNG
        
        result = image_denoiser.save_image(image_id, filename, format_type)
        
        if isinstance(result, dict) and 'error' in result:
            return jsonify(result), 400
        
        img_io, download_filename = result
        
        # Set the appropriate MIME type based on format
        mime_types = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'webp': 'image/webp'
        }
        
        mime_type = mime_types.get(format_type.lower(), 'image/png')
        
        # Using Flask 2.0+ parameters
        try:
            # Flask 2.0+ syntax
            return send_file(
                img_io,
                mimetype=mime_type,
                as_attachment=True,
                download_name=download_filename
            )
        except TypeError:
            # Fallback for older Flask versions
            return send_file(
                img_io,
                mimetype=mime_type,
                as_attachment=True,
                attachment_filename=download_filename
            )
    except Exception as e:
        print(f"Download error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Audio routes
@app.route('/api/audio/analyze', methods=['POST'])
def analyze_audio():
    """API endpoint to analyze an audio file for noise"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Check if filename is empty (happens when no file is selected)
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
            
        # Validate file extension (optional but helps with security)
        allowed_extensions = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
        filename = audio_file.filename.lower()
        if not any(filename.endswith(f'.{ext}') for ext in allowed_extensions):
            return jsonify({'error': f'Unsupported file format. Please upload a {", ".join(allowed_extensions)} file'}), 400
        
        # Process the audio file
        results = audio_denoiser.analyze_audio(audio_file)
        
        if 'error' in results:
            error_msg = results['error']
            print(f"Audio analysis error: {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        # Convert NumPy types to Python native types before JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        # Recursively convert all NumPy types in the results dictionary
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(v) for v in d]
            else:
                return convert_numpy(d)
        
        converted_results = convert_dict(results)
        return jsonify(converted_results)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Unexpected error in analyze_audio endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/audio/denoise', methods=['POST'])
def denoise_audio():
    """API endpoint to denoise an audio file"""
    try:
        print(f"Received denoise request with data: {request.json}")
        if not request.json:
            return jsonify({'error': 'No JSON data provided in request'}), 400
            
        if 'audio_id' not in request.json:
            return jsonify({'error': 'No audio ID provided'}), 400
        
        audio_id = request.json['audio_id']
        print(f"Processing denoise request for audio_id: {audio_id}")
        
        results = audio_denoiser.denoise_audio(audio_id)
        
        if 'error' in results:
            print(f"Audio denoising error: {results['error']}")
            return jsonify(results), 400
        
        # Convert NumPy types to Python native types before JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        # Recursively convert all NumPy types in the results dictionary
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(v) for v in d]
            else:
                return convert_numpy(d)
        
        converted_results = convert_dict(results)
        print(f"Successfully processed audio denoise request for ID: {audio_id}")
        return jsonify(converted_results)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Unexpected error in denoise_audio endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/audio/save/<audio_id>', methods=['GET'])
def save_audio(audio_id):
    """API endpoint to save/download a denoised audio file"""
    try:
        print(f"Received save audio request for ID: {audio_id}")
        
        filename = request.args.get('filename')
        format_type = request.args.get('format', 'wav')  # Default to WAV
        version = request.args.get('version', 'current')  # Default to current version
        
        print(f"Save parameters: filename={filename}, format={format_type}, version={version}")
        
        result = audio_denoiser.save_audio(audio_id, filename, format_type, version)
        
        if isinstance(result, dict) and 'error' in result:
            error_msg = result['error']
            print(f"Error saving audio: {error_msg}")
            return jsonify(result), 400
        
        audio_io, download_filename = result
        print(f"Successfully prepared file for download: {download_filename}")
        
        mime_type = 'audio/wav'
        if format_type.lower() == 'mp3':
            mime_type = 'audio/mpeg'
        
        # Create response with the file
        try:
            response = send_file(
                audio_io,
                mimetype=mime_type,
                as_attachment=True,
                download_name=download_filename
            )
            # Add headers to prevent caching
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            
            return response
        except Exception as e:
            print(f"Error sending file: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f"Error sending file: {str(e)}"}), 500
            
    except Exception as e:
        print(f"Error in save_audio endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/audio/preview/<audio_id>', methods=['GET'])
def preview_audio(audio_id):
    """API endpoint to get a base64 audio preview for the web player"""
    try:
        result = audio_denoiser.get_audio_preview(audio_id)
        
        if isinstance(result, dict) and 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 