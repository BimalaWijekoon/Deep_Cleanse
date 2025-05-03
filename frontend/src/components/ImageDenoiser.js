import React, { useState, useRef, useEffect } from 'react';
import '../styles/ImageDenoiser.css';
import { FaArrowLeft, FaUpload, FaSearch, FaMagic, FaSave, FaRedo, FaHatWizard, 
         FaDownload, FaInfoCircle, FaChartBar, FaChevronLeft, FaChevronRight, FaTimesCircle, 
         FaCheckCircle, FaCompressArrowsAlt, FaExpand, FaLock } from 'react-icons/fa';
import axios from 'axios';
import Modal from './Modal';

const ImageDenoiser = ({ goBack, playSound, isLoggedIn = false }) => {
  // State for managing the app
  const [originalImage, setOriginalImage] = useState(null);
  const [denoisedImage, setDenoisedImage] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [imageId, setImageId] = useState(null);
  const [status, setStatus] = useState('Ready to denoise your image');
  const [progress, setProgress] = useState(0);
  const [metrics, setMetrics] = useState(null);
  const [denoisingRound, setDenoisingRound] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState('');
  const [imageSize, setImageSize] = useState('');
  const [selectedImage, setSelectedImage] = useState(null);
  const [originalPreview, setOriginalPreview] = useState(null);
  const [denoisedPreview, setDenoisedPreview] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [sessionName, setSessionName] = useState('');
  
  // Modal state using our reusable Modal component
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalTitle, setModalTitle] = useState('');
  const [modalContent, setModalContent] = useState('');
  const [modalType, setModalType] = useState('info');
  const [modalButtons, setModalButtons] = useState([]);
  
  // Download modal state
  const [isDownloadModalOpen, setIsDownloadModalOpen] = useState(false);
  
  const [denoisingHistory, setDenoisingHistory] = useState([]);
  const [currentCompareIndex, setCurrentCompareIndex] = useState(0);
  const [showComparison, setShowComparison] = useState(false);
  const [fullscreenImage, setFullscreenImage] = useState(null);
  const [downloadFormat, setDownloadFormat] = useState('png');
  const [downloadRound, setDownloadRound] = useState(0);
  const fileInputRef = useRef(null);

  // Load session data if available in localStorage
  useEffect(() => {
    const sessionData = localStorage.getItem('currentImageSession');
    if (sessionData) {
      try {
        const session = JSON.parse(sessionData);
        loadExistingSession(session);
        
        // Clear the localStorage data after loading
        localStorage.removeItem('currentImageSession');
      } catch (error) {
        console.error('Error loading session data:', error);
      }
    }
  }, []);

  // Function to load an existing session
  const loadExistingSession = (session) => {
    setIsLoading(true);
    try {
      // Set session data
      setSessionId(session._id);
      setSessionName(session.sessionName);
      
      // Set original image
      if (session.originalImage) {
        setOriginalImage(session.originalImage);
        
        // Get image dimensions if possible
        const img = new Image();
        img.onload = () => {
          setImageSize(`${img.width} x ${img.height}`);
        };
        img.src = session.originalImage;
        
        // Extract file name from session name
        const nameParts = session.sessionName.split('_');
        if (nameParts.length >= 3) {
          setFileName(nameParts.slice(0, nameParts.length - 2).join('_'));
        } else {
          setFileName(session.sessionName);
        }
      }
      
      // Set analysis results if available
      if (session.analysisResults) {
        setAnalysisResults(session.analysisResults);
        setImageId(session.analysisResults.image_id);
        setStatus('Analysis complete. You can apply more denoising rounds.');
      }
      
      // Load denoising history
      const newHistory = [{
        round: 0,
        image: session.originalImage,
        metrics: null
      }];
      
      // Add denoised images to history
      if (session.denoisedImages && session.denoisedImages.length > 0) {
        session.denoisedImages.forEach(item => {
          newHistory.push({
            round: item.round,
            image: item.image,
            metrics: item.metrics
          });
        });
        
        // Set the latest denoised image and metrics
        const latestImage = session.denoisedImages[session.denoisedImages.length - 1];
        setDenoisedImage(latestImage.image);
        setMetrics(latestImage.metrics);
        setDenoisingRound(latestImage.round);
      }
      
      setDenoisingHistory(newHistory);
      setStatus(`Session "${session.sessionName}" loaded successfully!`);
      
    } catch (error) {
      console.error('Error loading session:', error);
      showMessageModal('Failed to load session data. Please try again.', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // Reset the application state
  const handleReset = () => {
    setOriginalImage(null);
    setDenoisedImage(null);
    setAnalysisResults(null);
    setImageId(null);
    setStatus('Ready to denoise your image');
    setProgress(0);
    setMetrics(null);
    setDenoisingRound(0);
    setIsLoading(false);
    setFileName('');
    setImageSize('');
    setSelectedImage(null);
    setOriginalPreview(null);
    setDenoisedPreview(null);
    setAnalysisResult(null);
    setErrorMsg('');
    setDenoisingHistory([]);
    setCurrentCompareIndex(0);
    setShowComparison(false);
    setFullscreenImage(null);
    setDownloadFormat('png');
    setDownloadRound(0);
    setIsDownloadModalOpen(false);
    setSessionId(null);
    setSessionName('');
  };

  // Handle file upload
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Check if file is an image
    if (!file.type.match('image.*')) {
      showMessageModal('Please select an image file (png, jpg, jpeg, etc.)', 'error');
      return;
    }

    setFileName(file.name);
    // Generate session name based on file name and user email
    const userEmail = localStorage.getItem('userEmail') || 'guest';
    const sanitizedFileName = file.name.replace(/\.[^/.]+$/, '').replace(/[^a-zA-Z0-9]/g, '_');
    setSessionName(`${sanitizedFileName}_${userEmail.split('@')[0]}_${Date.now()}`);
    
    // Get image dimensions
    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        setImageSize(`${img.width} x ${img.height}`);
        setOriginalImage(event.target.result);
        setStatus('Image loaded. Click "Analyze" to detect noise.');
        setProgress(0);
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  };

  // Show modal dialog using the Modal component
  const showMessageModal = (message, type = 'info', title = '') => {
    setModalContent(message);
    setModalType(type);
    setModalTitle(title || (type === 'error' ? 'Error' : type === 'success' ? 'Success' : 'Information'));
    setModalButtons([{ label: 'OK', onClick: () => setIsModalOpen(false) }]);
    setIsModalOpen(true);
  };

  // Show confirmation modal
  const showConfirmationModal = (message, onConfirm, title = 'Confirmation') => {
    setModalContent(message);
    setModalType('info');
    setModalTitle(title);
    setModalButtons([
      { label: 'No, Skip', onClick: () => setIsModalOpen(false), className: 'cancel-button' },
      { label: 'Yes, Continue', onClick: () => {
        setIsModalOpen(false);
        onConfirm();
      }}
    ]);
    setIsModalOpen(true);
  };

  // Show login required modal
  const showLoginModal = (message = 'Please log in to save denoised images. This feature is only available for registered users.') => {
    setModalContent(message);
    setModalType('error');
    setModalTitle('Login Required');
    setModalButtons([
      { label: 'Cancel', onClick: () => setIsModalOpen(false), className: 'cancel-button' },
      { label: 'Go to Login', onClick: () => {
        setIsModalOpen(false);
        goBack(); // Return to main menu where user can login
      }}
    ]);
    setIsModalOpen(true);
  };

  // Analyze image function
  const handleAnalyze = async () => {
    if (!originalImage || isLoading) return;
    
    try {
      setIsLoading(true);
      setStatus('Analyzing image for noise...');
      setProgress(10);
      
      const response = await axios.post('http://localhost:5001/api/image/analyze', {
        image: originalImage
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      setProgress(90);
      
      // Set the analysis results and imageId for future use
      setAnalysisResults(response.data);
      setImageId(response.data.image_id);
      
      setProgress(100);
      setStatus('Analysis complete. Click "Denoise" to process.');
      setIsLoading(false);

      // Save the original image for history
      if (denoisingHistory.length === 0) {
        setDenoisingHistory([{
          round: 0,
          image: originalImage,
          metrics: null
        }]);
      }
      
      // If user is logged in, start a new session with just the analysis results
      if (isLoggedIn) {
        try {
          const userEmail = localStorage.getItem('userEmail');
          if (userEmail) {
            // Create a new session with just analysis results
            const saveResponse = await axios.post('http://localhost:5000/api/image/save-session', {
              sessionName,
              userEmail,
              originalImage: originalImage,
              denoisedImages: [],
              analysisResults: response.data
            });
            
            if (saveResponse.data.sessionId) {
              setSessionId(saveResponse.data.sessionId);
              console.log('Analysis session started with ID:', saveResponse.data.sessionId);
            }
          }
        } catch (error) {
          console.error('Error saving analysis to database:', error);
        }
      }
    } catch (error) {
      console.error('Error analyzing image:', error);
      setStatus('Error during analysis. Please try again.');
      setProgress(0);
      setIsLoading(false);
      setErrorMsg(error.response?.data?.error || 'Error analyzing image');
      showMessageModal((error.response?.data?.error || 'Unknown error'), 'error', 'Error Analyzing Image');
    }
  };
  
  // Denoise image function
  const handleDenoise = async () => {
    if (!imageId || isLoading) return;
    
    // Check if user has already done one round and is not logged in
    if (!isLoggedIn && denoisingRound >= 1) {
      showLoginModal('Please log in to use additional denoising rounds. This feature is only available for registered users.');
      return;
    }
    
    try {
      setIsLoading(true);
      setStatus(`Starting denoising round ${denoisingRound + 1}...`);
      setProgress(10);
      
      const response = await axios.post('http://localhost:5001/api/image/denoise', {
        image_id: imageId
      });
      
      // Update the progress and state
      setProgress(90);
      setDenoisedImage(response.data.denoised_image);
      setMetrics(response.data.metrics);
      setDenoisingRound(response.data.denoising_round);
      
      // Add to history
      const newDenoisingHistory = [
        ...denoisingHistory,
        {
          round: response.data.denoising_round,
          image: response.data.denoised_image,
          metrics: response.data.metrics
        }
      ];
      
      setDenoisingHistory(newDenoisingHistory);
      
      // Save session to database if user is logged in
      if (isLoggedIn) {
        try {
          const userEmail = localStorage.getItem('userEmail');
          if (userEmail) {
            // Save denoised image session to database with all available metrics
            const saveResponse = await axios.post('http://localhost:5000/api/image/save-session', {
              sessionName,
              userEmail,
              originalImage: denoisingHistory[0].image,
              denoisedImages: newDenoisingHistory.slice(1).map(item => ({
                round: item.round,
                image: item.image,
                metrics: {
                  // Include all available metrics from the response
                  ...item.metrics,
                  psnr: item.metrics.psnr,
                  ssim: item.metrics.ssim,
                  var_reduction: item.metrics.var_reduction,
                  entropy_reduction: item.metrics.entropy_reduction,
                  noise_level: analysisResults?.overall_noise_level || 0,
                  noise_types: analysisResults?.noise_types || [],
                  low_improvement: item.metrics.var_reduction < 5,
                  overall_noise_level: analysisResults?.overall_noise_level || 0,
                  noise_levels: analysisResults?.noise_levels || {}
                }
              })),
              analysisResults: analysisResults
            });
            
            if (saveResponse.data.sessionId) {
              setSessionId(saveResponse.data.sessionId);
              console.log('Session saved with ID:', saveResponse.data.sessionId);
            }
          }
        } catch (error) {
          console.error('Error saving session to database:', error);
        }
      }
      
      setProgress(100);
      if (!isLoggedIn) {
        setStatus(`Denoising complete! Login required for additional rounds.`);
      } else {
        setStatus(`Denoising round ${response.data.denoising_round} complete! You can save or apply another round.`);
      }
      setIsLoading(false);
      
      // Only show confirmation for additional rounds if the user is logged in
      if (isLoggedIn) {
        // Ask if user wants another round after the first round
        if (response.data.denoising_round >= 1 && response.data.metrics.var_reduction < 5) {
          setTimeout(() => {
            showConfirmationModal('Limited improvement detected. Do you still want to apply another round of denoising?', handleDenoise, 'Continue Denoising?');
          }, 500);
        } else if (response.data.denoising_round >= 1) {
          setTimeout(() => {
            showConfirmationModal('Do you want to apply another round of denoising to improve the image further?', handleDenoise, 'Continue Denoising?');
          }, 500);
        }
      }
    } catch (error) {
      console.error('Error denoising image:', error);
      setStatus('Error during denoising. Please try again.');
      setProgress(0);
      setIsLoading(false);
      setErrorMsg(error.response?.data?.error || 'Error denoising image');
      showMessageModal((error.response?.data?.error || 'Unknown error'), 'error', 'Error Denoising Image');
    }
  };
  
  // Save image function
  const handleSave = () => {
    if (!imageId) return;
    
    // Check if user is logged in
    if (!isLoggedIn) {
      showLoginModal();
      return;
    }
    
    setIsDownloadModalOpen(true);
  };

  const renderNoiseTypes = (types) => {
    if (!types || types.length === 0) return 'None detected';
    
    return types.map(type => {
      switch(type) {
        case 'salt_pepper':
          return 'Salt & Pepper';
        case 'gaussian':
          return 'Gaussian';
        case 'speckle':
          return 'Speckle';
        default:
          return type;
      }
    }).join(', ');
  };

  const renderNoiseLevel = (level) => {
    if (level === undefined) return 'N/A';
    
    // Format as percentage with one decimal place
    return `${level.toFixed(1)}%`;
  };

  const handleBackClick = () => {
    goBack();
  };

  const toggleComparison = () => {
    setShowComparison(!showComparison);
    setCurrentCompareIndex(0);
  };

  const handlePrevImage = () => {
    setCurrentCompareIndex(prev => 
      prev > 0 ? prev - 1 : denoisingHistory.length - 1
    );
  };

  const handleNextImage = () => {
    setCurrentCompareIndex(prev => 
      prev < denoisingHistory.length - 1 ? prev + 1 : 0
    );
  };

  const handleFullscreen = (imageSrc) => {
    setFullscreenImage(imageSrc);
  };

  const closeFullscreen = () => {
    setFullscreenImage(null);
  };

  const downloadImage = () => {
    if (denoisingHistory.length === 0) return;
    
    const timestamp = new Date().toISOString().replace(/:/g, '-').replace(/\..+/, '');
    const filename = `denoised_${timestamp}`;
    const fullFilename = `${filename}.${downloadFormat}`;
    
    const downloadUrl = `http://localhost:5001/api/image/save/${imageId}?filename=${filename}&format=${downloadFormat}`;
    
    // Create a temporary link element and trigger download
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = fullFilename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    showMessageModal(`Denoised image successfully downloaded as ${downloadFormat.toUpperCase()}!`, 'success', 'Download Complete');
    setIsDownloadModalOpen(false);
  };

  // Download modal content component
  const DownloadModalContent = () => (
    <>
              <div className="image-download-options">
                <div className="image-option-group">
                  <label>Choose Version:</label>
                  <select 
                    className="image-download-select"
                    value={downloadRound}
                    onChange={(e) => setDownloadRound(parseInt(e.target.value))}
                  >
                    {denoisingHistory.slice(1).map((item, index) => (
                      <option key={index + 1} value={index + 1}>
                        Round {index + 1} {index + 1 === denoisingRound ? '(Latest)' : ''}
                      </option>
                    ))}
                  </select>
                </div>
                
                <div className="image-option-group">
                  <label>File Format:</label>
                  <div className="image-format-options">
                    <button 
                      className={`image-format-button ${downloadFormat === 'png' ? 'image-active' : ''}`}
                      onClick={() => setDownloadFormat('png')}
                    >
                      PNG Format
                    </button>
                    <button 
                      className={`image-format-button ${downloadFormat === 'jpg' ? 'image-active' : ''}`}
                      onClick={() => setDownloadFormat('jpg')}
                    >
                      JPG Format
                    </button>
                    <button 
                      className={`image-format-button ${downloadFormat === 'webp' ? 'image-active' : ''}`}
                      onClick={() => setDownloadFormat('webp')}
                    >
                      WEBP Format
                    </button>
                  </div>
                </div>
              </div>
    </>
  );

  return (
    <div className="image-denoiser-container">
      {/* Fullscreen Image Viewer */}
      {fullscreenImage && (
        <div className="image-fullscreen-overlay" onClick={closeFullscreen}>
          <div className="image-fullscreen-content">
            <button className="image-close-fullscreen" onClick={closeFullscreen}>
              <FaCompressArrowsAlt />
                </button>
            <img src={fullscreenImage} alt="Fullscreen view" />
          </div>
        </div>
      )}

      {/* Modal dialogs using reusable Modal component */}
      <Modal 
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title={modalTitle}
        type={modalType}
        buttons={modalButtons}
      >
        {modalContent}
      </Modal>

      {/* Download Modal Dialog */}
      <Modal
        isOpen={isDownloadModalOpen}
        onClose={() => setIsDownloadModalOpen(false)}
        title="Download Image"
        type="default"
        buttons={[
          {
            label: 'Download',
            onClick: downloadImage,
            icon: <FaDownload />
          }
        ]}
      >
        <DownloadModalContent />
      </Modal>

      <div className="image-glass-panel glass-panel">
        <div className="image-header-section">
          <button className="image-back-button" onClick={handleBackClick}>
            <FaArrowLeft /> Back
          </button>
          <h1>Image Denoiser</h1>
          <button 
            className={`image-compare-button ${showComparison ? 'image-active' : ''}`}
            onClick={toggleComparison}
            disabled={denoisingHistory.length < 2}
          >
            <FaChartBar /> Compare Results
          </button>
        </div>

        {showComparison ? (
          <div className="image-comparison-view">
            <div className="image-comparison-controls">
              <h3>Denoising History (Round {currentCompareIndex} of {denoisingHistory.length - 1})</h3>
              <div>
                <button 
                  className="image-nav-button"
                  onClick={handlePrevImage}
                  disabled={currentCompareIndex <= 1}
                >
                  <FaChevronLeft /> Previous
                </button>
                <button 
                  className="image-nav-button"
                  onClick={handleNextImage}
                  disabled={currentCompareIndex >= denoisingHistory.length - 1}
                >
                  Next <FaChevronRight />
                </button>
              </div>
            </div>
            
            <div className="image-comparison-images">
              <div className="image-comparison-image-card">
                <h4>Original Image</h4>
                <img 
                  src={denoisingHistory[0]?.image} 
                  alt="Original" 
                  onClick={() => handleFullscreen(denoisingHistory[0]?.image)}
                />
              </div>
              
              <div className="image-comparison-image-card">
                <h4>Round {currentCompareIndex} Result</h4>
                <img 
                  src={denoisingHistory[currentCompareIndex]?.image} 
                  alt={`Round ${currentCompareIndex}`}
                  onClick={() => handleFullscreen(denoisingHistory[currentCompareIndex]?.image)}
                />
              </div>
            </div>
            
            <div className="image-comparison-stats">
              <h4>Quality Improvement</h4>
              <div className="image-stats-grid">
                <div className="image-stat-item">
                  <span className="image-stat-label">PSNR:</span>
                  <span className="image-stat-value">
                    {denoisingHistory[currentCompareIndex]?.metrics?.psnr.toFixed(2)} dB
                  </span>
                </div>
                <div className="image-stat-item">
                  <span className="image-stat-label">SSIM:</span>
                  <span className="image-stat-value">
                    {denoisingHistory[currentCompareIndex]?.metrics?.ssim.toFixed(4)}
                  </span>
                </div>
                <div className="image-stat-item">
                  <span className="image-stat-label">Noise Reduction:</span>
                  <span className="image-stat-value">
                    {denoisingHistory[currentCompareIndex]?.metrics?.var_reduction.toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="image-content-section">
            <div className="image-sidebar">
              <div className="image-upload-section glass-card">
                <h3>Upload Image</h3>
                <button 
                  className="image-upload-button"
                  onClick={() => fileInputRef.current.click()}
                  disabled={isLoading}
                >
                  <FaUpload /> Select Image
                </button>
                <input 
                  type="file" 
                  ref={fileInputRef}
                  accept="image/*" 
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                />
                
                {fileName && (
                  <div className="image-file-info">
                    <p><strong>File:</strong> {fileName}</p>
                    <p><strong>Size:</strong> {imageSize}</p>
                  </div>
                )}
              </div>

              <div className="image-analysis-section glass-card">
                <h3>Noise Analysis</h3>
                <div className="image-analysis-content">
                  {analysisResults ? (
                    <>
                      {analysisResults.has_noise ? (
                        <>
                          <div className="image-analysis-header">
                            <h4>NOISE DETECTED</h4>
                            <div className="image-noise-meter-container">
                              <div className="image-noise-meter">
                                <div 
                                  className="image-noise-meter-fill" 
                                  style={{ 
                                    width: `${Math.min(100, analysisResults.overall_noise_level * 10)}%`,
                                    backgroundColor: analysisResults.overall_noise_level > 5 
                                      ? '#ff4d4d' 
                                      : analysisResults.overall_noise_level > 2 
                                        ? '#ffa64d' 
                                        : '#4dff88'
                                  }}
                                ></div>
                              </div>
                              <p className="image-noise-level">
                                {analysisResults.overall_noise_level.toFixed(2)}%
                              </p>
                            </div>
                          </div>

                          <div className="image-noise-types">
                            <h4>Detected Noise Types:</h4>
                            <ul>
                              {analysisResults.noise_types.map((type) => (
                                <li key={type}>
                                  {type.replace('_', ' ').charAt(0).toUpperCase() + type.replace('_', ' ').slice(1)}: 
                                  <div className="image-noise-type-meter">
                                    <div 
                                      className="image-noise-type-fill" 
                                      style={{ 
                                        width: `${Math.min(100, analysisResults.noise_levels[type] * 10)}%`,
                                        backgroundColor: analysisResults.noise_levels[type] > 5 
                                          ? '#ff4d4d' 
                                          : analysisResults.noise_levels[type] > 2 
                                            ? '#ffa64d' 
                                            : '#4dff88'
                                      }}
                                    ></div>
                                  </div>
                                  {' '}{analysisResults.noise_levels[type].toFixed(2)}%
                                </li>
                              ))}
                            </ul>
                          </div>

                          <div className="image-recommendations">
                            <h4>Recommendation:</h4>
                            <p>{analysisResults.recommendations}</p>
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="image-analysis-header">
                            <h4>NO SIGNIFICANT NOISE DETECTED</h4>
                          </div>
                          <p>The image appears to be clear with minimal noise.</p>
                          <p>You can still try denoising if you want to enhance image quality further.</p>
                        </>
                      )}
                    </>
                  ) : (
                    <p className="image-empty-message">Upload and analyze an image to see noise detection results.</p>
                  )}
                </div>
              </div>

              <div className="image-actions-section glass-card">
                <h3>Actions</h3>
                <button 
                  className="image-action-button image-analyze-button"
                  onClick={handleAnalyze}
                  disabled={!originalImage || isLoading}
                >
                  <FaSearch /> Analyze Image
                </button>
                
                <button
                  className={`image-action-button image-denoise-button ${!isLoggedIn && denoisingRound >= 1 ? 'image-disabled-button' : ''}`}
                  onClick={handleDenoise}
                  disabled={!imageId || isLoading}
                  title={!isLoggedIn && denoisingRound >= 1 ? "Login required for additional denoising rounds" : "Denoise Image"}
                >
                  {!isLoggedIn && denoisingRound >= 1 && <FaLock className="image-lock-icon" />} <FaMagic /> Denoise Image
                </button>
                
                <button
                  className={`image-action-button image-save-button ${!isLoggedIn ? 'image-disabled-button' : ''}`}
                  onClick={handleSave}
                  disabled={!denoisedImage || isLoading}
                  title={!isLoggedIn ? "Login required to save results" : "Save result"}
                >
                  {!isLoggedIn && <FaLock className="image-lock-icon" />} <FaDownload /> Save Result
                </button>
                
                <button 
                  className="image-action-button image-reset-button"
                  onClick={handleReset}
                  disabled={isLoading}
                >
                  <FaRedo /> Reset
                </button>
              </div>
            </div>

            <div className="image-main-content">
              <div className="image-status-bar glass-card">
                <div className="image-progress-container">
                  <div 
                    className="image-progress-bar" 
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
                <p className="image-status-text">{status}</p>
                {isLoading && <div className="image-loading-spinner"></div>}
              </div>

              <div className="image-display-section">
                <div className="image-panel glass-card">
                  <h3>Original Image</h3>
                  <div className="image-wrapper">
                    {originalImage ? (
                      <div className="image-container">
                        <img 
                          src={originalImage} 
                          alt="Original" 
                          className="image-display-image"
                          onClick={() => handleFullscreen(originalImage)}
                        />
                        <button 
                          className="image-fullscreen-button" 
                          onClick={() => handleFullscreen(originalImage)}
                        >
                          <FaExpand />
                        </button>
                      </div>
                    ) : (
                      <div className="image-no-image">
                        <p>No image loaded</p>
                      </div>
                    )}
                  </div>
                </div>

                <div className="image-panel glass-card">
                  <h3>Denoised Image</h3>
                  <div className="image-wrapper">
                    {denoisedImage ? (
                      <div className="image-container">
                        <img 
                          src={denoisedImage} 
                          alt="Denoised" 
                          className="image-display-image"
                          onClick={() => handleFullscreen(denoisedImage)}
                        />
                        <button 
                          className="image-fullscreen-button" 
                          onClick={() => handleFullscreen(denoisedImage)}
                        >
                          <FaExpand />
                        </button>
                      </div>
                    ) : (
                      <div className="image-no-image">
                        <p>No processed image yet</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="image-stats-section glass-card">
                <h3>Image Statistics</h3>
                {denoisingHistory.length > 1 ? (
                  <div className="image-stats-content">
                    {denoisingHistory.slice(1).map((historyItem, index) => (
                      <div key={`round-${index + 1}`} className="image-stats-round">
                        <div className="image-round-badge">Round {index + 1}</div>
                        
                        <div className="image-metrics">
                          <div className="image-metric">
                            <span className="image-metric-name">PSNR:</span>
                            <span className="image-metric-value">{historyItem.metrics?.psnr.toFixed(2)} dB</span>
                            <div className="image-metric-bar-container">
                              <div 
                                className="image-metric-bar" 
                                style={{ width: `${Math.min(100, historyItem.metrics?.psnr * 3)}%` }}
                              ></div>
                            </div>
                            <span className="image-metric-description">Higher is better</span>
                          </div>
                          
                          <div className="image-metric">
                            <span className="image-metric-name">SSIM:</span>
                            <span className="image-metric-value">{historyItem.metrics?.ssim.toFixed(4)}</span>
                            <div className="image-metric-bar-container">
                              <div 
                                className="image-metric-bar" 
                                style={{ width: `${Math.min(100, historyItem.metrics?.ssim * 100)}%` }}
                              ></div>
                            </div>
                            <span className="image-metric-description">Higher is better</span>
                          </div>
                          
                          <div className="image-metric">
                            <span className="image-metric-name">Noise Reduction:</span>
                            <span className="image-metric-value">{historyItem.metrics?.var_reduction.toFixed(2)}%</span>
                            <div className="image-metric-bar-container">
                              <div 
                                className="image-metric-bar" 
                                style={{ width: `${Math.min(100, historyItem.metrics?.var_reduction)}%` }}
                              ></div>
                            </div>
                          </div>
                        </div>
                        
                        {historyItem.metrics?.low_improvement && index > 0 && (
                          <p className="image-improvement-note">
                            <FaInfoCircle /> Limited improvement from previous round. Further denoising may not yield significant results.
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="image-empty-message">Process an image to see quality metrics.</p>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageDenoiser; 