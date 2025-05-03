# DeepCleanse - Advanced Media Denoising

This is the frontend application for DeepCleanse, a full-stack solution for advanced media denoising using deep learning.

## Features

- **Image Denoising**: Remove noise from images and enhance quality
  - Support for multiple noise types (salt & pepper, gaussian, speckle)
  - Real-time analysis and metrics
  - Multiple rounds of progressive denoising

- **Audio Denoising**: Clean up audio recordings and improve sound quality
  - Background noise removal
  - Real-time audio preview
  - Signal-to-noise ratio improvement metrics

## Setup Instructions

### Prerequisites
- Node.js 14+
- Python 3.7+
- Required Python packages: `flask`, `flask-cors`, `tensorflow`, `pillow`, `numpy`, `opencv-python`, `librosa`, `soundfile`

### Installation

1. Install frontend dependencies:
```
npm install
```

2. Install backend dependencies:
```
pip install flask flask-cors tensorflow pillow numpy opencv-python librosa soundfile
```

### Running the Application

To start both the frontend and backend simultaneously:
```
npm start
```

The application will be available at [http://localhost:3000](http://localhost:3000)

## Architecture

- **Frontend**: React.js with modern UI components
- **Backend**: Python Flask API with TensorFlow models
- **API**: RESTful endpoints for image and audio processing

## Denoising Process

1. Upload media file (image or audio)
2. Analyze for noise types and patterns
3. Apply deep learning models to remove noise
4. Compare original and denoised media
5. Save enhanced results

## API Endpoints

- `POST /api/analyze` - Analyze an image for noise
- `POST /api/denoise` - Denoise an image
- `GET /api/save/<image_id>` - Download a denoised image

## Building for Production

To create a production build:

```
npm run build
```

The build folder will contain the compiled application ready for deployment. 