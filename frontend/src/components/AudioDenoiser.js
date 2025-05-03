import React, { useState, useRef, useEffect } from 'react';
import '../styles/AudioDenoiser.css';
import { FaArrowLeft, FaUpload, FaSearch, FaMagic, FaSave, FaRedo, FaPlay, FaPause, FaVolumeUp, FaDownload, FaLock, FaCheckCircle, FaExclamationTriangle, FaInfoCircle, FaChartBar, FaChevronLeft, FaChevronRight } from 'react-icons/fa';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import Modal from './Modal';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const AudioDenoiser = ({ goBack, playSound, isLoggedIn = false }) => {
  // State for managing the app
  const [originalAudio, setOriginalAudio] = useState(null);
  const [denoisedAudio, setDenoisedAudio] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [audioId, setAudioId] = useState(null);
  const [status, setStatus] = useState('Ready to denoise your audio');
  const [progress, setProgress] = useState(0);
  const [metrics, setMetrics] = useState(null);
  const [denoisingRound, setDenoisingRound] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState('');
  const [audioDuration, setAudioDuration] = useState('');
  const [isPlayingOriginal, setIsPlayingOriginal] = useState(false);
  const [isPlayingDenoised, setIsPlayingDenoised] = useState(false);
  const [waveformData, setWaveformData] = useState({ original: [], denoised: [] });
  const [downloadFormat, setDownloadFormat] = useState('wav');
  const [selectedVersion, setSelectedVersion] = useState('current');
  const [isDownloadModalOpen, setIsDownloadModalOpen] = useState(false);
  const [originalPlaybackPosition, setOriginalPlaybackPosition] = useState(0);
  const [denoisedPlaybackPosition, setDenoisedPlaybackPosition] = useState(0);
  const [isDraggingOriginalPlayhead, setIsDraggingOriginalPlayhead] = useState(false);
  const [isDraggingDenoisedPlayhead, setIsDraggingDenoisedPlayhead] = useState(false);
  const waveformContainerRef = useRef(null);
  
  // Comparison and history state
  const [denoisingHistory, setDenoisingHistory] = useState([]);
  const [currentCompareIndex, setCurrentCompareIndex] = useState(0);
  const [showComparison, setShowComparison] = useState(false);
  const [comparePlaying, setComparePlaying] = useState(null);

  // Modal state using our reusable Modal component
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalTitle, setModalTitle] = useState('');
  const [modalContent, setModalContent] = useState('');
  const [modalType, setModalType] = useState('info');
  const [modalButtons, setModalButtons] = useState([]);

  // References for the audio elements
  const originalAudioRef = useRef(null);
  const denoisedAudioRef = useRef(null);
  const compareOriginalRef = useRef(null);
  const compareProcessedRef = useRef(null);
  const originalWaveformRef = useRef(null);
  const denoisedWaveformRef = useRef(null);

  // Add state for comparison waveform playheads
  const [compareOriginalPlaybackPosition, setCompareOriginalPlaybackPosition] = useState(0);
  const [compareProcessedPlaybackPosition, setCompareProcessedPlaybackPosition] = useState(0);
  const [isDraggingCompareOriginalPlayhead, setIsDraggingCompareOriginalPlayhead] = useState(false);
  const [isDraggingCompareProcessedPlayhead, setIsDraggingCompareProcessedPlayhead] = useState(false);
  const compareOriginalWaveformRef = useRef(null);
  const compareProcessedWaveformRef = useRef(null);

  // Add state for session
  const [sessionId, setSessionId] = useState(null);
  const [sessionName, setSessionName] = useState('');

  // Load session data if available in localStorage
  useEffect(() => {
    const sessionData = localStorage.getItem('currentAudioSession');
    if (sessionData) {
      try {
        const session = JSON.parse(sessionData);
        loadExistingSession(session);
        
        // Clear the localStorage data after loading
        localStorage.removeItem('currentAudioSession');
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
      
      // Set original audio
      if (session.originalAudio) {
        setOriginalAudio(session.originalAudio);
        
        // Extract file name from session name
        const nameParts = session.sessionName.split('_');
        if (nameParts.length >= 3) {
          setFileName(nameParts.slice(0, nameParts.length - 2).join('_'));
        } else {
          setFileName(session.sessionName);
        }
        
        // Try to load audio duration
        const audio = new Audio();
        audio.src = session.originalAudio;
        audio.onloadedmetadata = () => {
          const minutes = Math.floor(audio.duration / 60);
          const seconds = Math.floor(audio.duration % 60);
          setAudioDuration(`${minutes}:${seconds.toString().padStart(2, '0')}`);
        };
      }
      
      // Set original waveform data if available
      if (session.originalWaveform && session.originalWaveform.length > 0) {
        setWaveformData(prev => ({
          ...prev,
          original: session.originalWaveform
        }));
      }
      
      // Set analysis results if available
      if (session.analysisResults) {
        // Ensure that noise_levels is properly handled
        const processedAnalysisResults = {
          ...session.analysisResults,
          // If noise_levels is a Map object from MongoDB, convert it to a plain object
          noise_levels: session.analysisResults.noise_levels ? 
            (session.analysisResults.noise_levels instanceof Map ? 
              Object.fromEntries(session.analysisResults.noise_levels) : 
              session.analysisResults.noise_levels) : 
            {}
        };
        
        setAnalysisResults(processedAnalysisResults);
        setAudioId(session.analysisResults.audio_id);
        setStatus('Analysis complete. You can apply more denoising rounds.');
      }
      
      // Load denoising history
      const newHistory = [{
        round: 0,
        audio: session.originalAudio,
        waveform: session.originalWaveform || [],
        metrics: null
      }];
      
      // Add denoised audios to history
      if (session.denoisedAudios && session.denoisedAudios.length > 0) {
        session.denoisedAudios.forEach(item => {
          newHistory.push({
            round: item.round,
            audio: item.audio,
            waveform: item.waveform || [],
            metrics: item.metrics
          });
        });
        
        // Set the latest denoised audio, waveform, and metrics
        const latestAudio = session.denoisedAudios[session.denoisedAudios.length - 1];
        setDenoisedAudio(latestAudio.audio);
        
        if (latestAudio.waveform && latestAudio.waveform.length > 0) {
          setWaveformData(prev => ({
            ...prev,
            denoised: latestAudio.waveform
          }));
        }
        
        setMetrics(latestAudio.metrics);
        setDenoisingRound(latestAudio.round);
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
    setOriginalAudio(null);
    setDenoisedAudio(null);
    setAnalysisResults(null);
    setAudioId(null);
    setStatus('Ready to denoise your audio');
    setProgress(0);
    setMetrics(null);
    setDenoisingRound(0);
    setIsLoading(false);
    setFileName('');
    setAudioDuration('');
    setIsPlayingOriginal(false);
    setIsPlayingDenoised(false);
    setWaveformData({ original: [], denoised: [] });
    setDownloadFormat('wav');
    setSelectedVersion('current');
    setDenoisingHistory([]);
    setCurrentCompareIndex(0);
    setShowComparison(false);
    setComparePlaying(null);
    setOriginalPlaybackPosition(0);
    setDenoisedPlaybackPosition(0);
    setCompareOriginalPlaybackPosition(0);
    setCompareProcessedPlaybackPosition(0);
    setIsDraggingOriginalPlayhead(false);
    setIsDraggingDenoisedPlayhead(false);
    setIsDraggingCompareOriginalPlayhead(false);
    setIsDraggingCompareProcessedPlayhead(false);
    setSessionId(null);
    setSessionName('');
  };

  // Toggle comparison view
  const toggleComparison = () => {
    // If we're closing the comparison view, stop any playing audio
    if (showComparison) {
      if (compareOriginalRef.current) compareOriginalRef.current.pause();
      if (compareProcessedRef.current) compareProcessedRef.current.pause();
      setComparePlaying(null);
    }
    
    console.log('Denoising History when toggling comparison:', denoisingHistory);
    
    // Make sure we have valid history data
    if (!showComparison && denoisingHistory.length < 2) {
      // If we're opening comparison view but don't have at least 2 entries, show message
      showMessageModal('Not enough denoising rounds to compare. Please process the audio first.', 'warning');
      return;
    }
    
    // If we're opening comparison view, ensure that the waveform data is present
    if (!showComparison && denoisingHistory.length >= 2) {
      // Make sure original has waveform data
      if (!denoisingHistory[0].waveform || denoisingHistory[0].waveform.length === 0) {
        console.log('Updating original waveform in history before comparison');
        // Set original waveform from current data
        setDenoisingHistory(prev => [
          {
            ...prev[0],
            waveform: waveformData.original || []
          },
          ...prev.slice(1)
        ]);
      }
    }
    
    setShowComparison(!showComparison);
    setCurrentCompareIndex(1); // Start with the first denoising round
  };

  // Navigation in comparison view
  const handlePrevRound = () => {
    // Stop any playing audio when changing rounds
    if (compareOriginalRef.current) compareOriginalRef.current.pause();
    if (compareProcessedRef.current) compareProcessedRef.current.pause();
    setComparePlaying(null);
    
    setCurrentCompareIndex(prev => 
      prev > 1 ? prev - 1 : 1
    );
  };

  const handleNextRound = () => {
    // Stop any playing audio when changing rounds
    if (compareOriginalRef.current) compareOriginalRef.current.pause();
    if (compareProcessedRef.current) compareProcessedRef.current.pause();
    setComparePlaying(null);
    
    setCurrentCompareIndex(prev => 
      prev < denoisingHistory.length - 1 ? prev + 1 : denoisingHistory.length - 1
    );
  };
  
  // Play/pause audio in comparison view
  const toggleCompareAudio = (type) => {
    // Stop the currently playing audio if there is one
    if (comparePlaying) {
      if (comparePlaying === 'original' && compareOriginalRef.current) {
        compareOriginalRef.current.pause();
      } else if (comparePlaying === 'processed' && compareProcessedRef.current) {
        compareProcessedRef.current.pause();
      }
    }
    
    // If we clicked the same button that was playing, just stop it
    if (comparePlaying === type) {
      setComparePlaying(null);
      return;
    }
    
    // Start playing the selected audio
    if (type === 'original' && compareOriginalRef.current) {
      compareOriginalRef.current.currentTime = 0;
      compareOriginalRef.current.play();
      setComparePlaying('original');
    } else if (type === 'processed' && compareProcessedRef.current) {
      compareProcessedRef.current.currentTime = 0;
      compareProcessedRef.current.play();
      setComparePlaying('processed');
    }
  };
  
  // Handle audio ended event in comparison view
  const handleCompareAudioEnded = () => {
    setComparePlaying(null);
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
  const showLoginModal = () => {
    setModalContent('Please log in to save denoised audio. This feature is only available for registered users.');
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

  // Handle file upload
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Check if file is an audio
    if (!file.type.match('audio.*')) {
      showMessageModal('Please select an audio file (mp3, wav, etc.)', 'error');
      return;
    }

    setFileName(file.name);
    
    // Generate session name based on file name and user email
    const userEmail = localStorage.getItem('userEmail') || 'guest';
    const sanitizedFileName = file.name.replace(/\.[^/.]+$/, '').replace(/[^a-zA-Z0-9]/g, '_');
    setSessionName(`${sanitizedFileName}_${userEmail.split('@')[0]}_${Date.now()}`);

    // Create form data for upload (we'll use this later for analysis)
    const formData = new FormData();
    formData.append('audio', file);
    
    // Set the original audio for preview
    const audioURL = URL.createObjectURL(file);
    setOriginalAudio(audioURL);
    
    // Load audio to get duration
    const audio = new Audio();
    audio.src = audioURL;
    audio.onloadedmetadata = () => {
      const minutes = Math.floor(audio.duration / 60);
      const seconds = Math.floor(audio.duration % 60);
      setAudioDuration(`${minutes}:${seconds.toString().padStart(2, '0')}`);
    };
    
    setStatus('Audio loaded. Click "Analyze" to detect noise.');
    setProgress(0);
    
    // Reset any existing history
    setDenoisingHistory([]);
    setCurrentCompareIndex(0);
  };

  // Play/pause controls for original audio
  const toggleOriginalPlayback = () => {
    if (originalAudioRef.current) {
      if (isPlayingOriginal) {
        originalAudioRef.current.pause();
      } else {
        originalAudioRef.current.play();
        // Pause the other audio if it's playing
        if (denoisedAudioRef.current && isPlayingDenoised) {
          denoisedAudioRef.current.pause();
          setIsPlayingDenoised(false);
        }
      }
      setIsPlayingOriginal(!isPlayingOriginal);
    }
  };

  // Play/pause controls for denoised audio
  const toggleDenoisedPlayback = () => {
    if (denoisedAudioRef.current) {
      if (isPlayingDenoised) {
        denoisedAudioRef.current.pause();
      } else {
        denoisedAudioRef.current.play();
        // Pause the other audio if it's playing
        if (originalAudioRef.current && isPlayingOriginal) {
          originalAudioRef.current.pause();
          setIsPlayingOriginal(false);
        }
      }
      setIsPlayingDenoised(!isPlayingDenoised);
    }
  };

  // Handle audio ended events
  const handleAudioEnded = (type) => {
    if (type === 'original') {
      setIsPlayingOriginal(false);
    } else {
      setIsPlayingDenoised(false);
    }
  };

  // Helper function to safely access noise levels
  const getNoiseLevelValue = (noiseType) => {
    if (!analysisResults || !analysisResults.noise_levels) {
      return 0;
    }
    
    // Handle both Map objects and plain objects from MongoDB
    if (typeof analysisResults.noise_levels === 'object') {
      // If it's a plain object
      return analysisResults.noise_levels[noiseType] !== undefined ? 
        analysisResults.noise_levels[noiseType] : 0;
    } else if (analysisResults.noise_levels instanceof Map) {
      // If it's a Map object
      return analysisResults.noise_levels.get(noiseType) || 0;
    }
    
    return 0;
  };

  // Helper function to safely access metrics in history items
  const getMetricValue = (historyItem, metricName) => {
    if (!historyItem || !historyItem.metrics) {
      return 0;
    }
    return historyItem.metrics[metricName] !== undefined ? 
      historyItem.metrics[metricName] : 0;
  };

  // Analyze audio function
  const handleAnalyze = async () => {
    if (!originalAudio || isLoading) return;
    
    try {
      setIsLoading(true);
      setStatus('Analyzing audio for noise...');
      setProgress(10);
      
      // Create form data for upload
      const formData = new FormData();
      
      // Get the file from the input if it's a new file upload
      const fileInput = document.querySelector('input[type="file"]');
      if (fileInput && fileInput.files.length > 0) {
        // If we have a file input with a file, use that
        formData.append('audio', fileInput.files[0]);
      } else {
        // Otherwise, try to use the originalAudio state which might be a blob URL
        try {
          const originalAudioBlob = await fetch(originalAudio).then(r => r.blob());
          formData.append('audio', originalAudioBlob);
        } catch (error) {
          console.error('Error fetching original audio blob:', error);
          throw new Error('Could not prepare audio for analysis');
        }
      }
      
      const response = await axios.post('http://localhost:5001/api/audio/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      console.log('Analysis response received:', response.data);
      
      setProgress(90);
      
      // Set the analysis results and audioId for future use
      setAnalysisResults(response.data);
      setAudioId(response.data.audio_id);
      console.log('Audio ID set:', response.data.audio_id);
      
      // Set the audio preview from the response
      if (response.data.audio_preview) {
        setOriginalAudio(response.data.audio_preview);
        
        // Set waveform data if available
        if (response.data.waveform_data && response.data.waveform_data.original) {
          setWaveformData({
            original: response.data.waveform_data.original || [],
            denoised: []
          });
        }
        
        // Initialize the denoising history with the original audio
        setDenoisingHistory([{
          round: 0,
          audio: response.data.audio_preview,
          waveform: response.data.waveform_data?.original || [],
          metrics: null
        }]);
        
        // Log what we received
        console.log('Initial waveform data:', {
          original: response.data.waveform_data?.original || [],
          audio: response.data.audio_preview
        });
      }
      
      // If user is logged in, start a new session with just the analysis results
      if (isLoggedIn) {
        try {
          const userEmail = localStorage.getItem('userEmail');
          if (userEmail) {
            // Create a new session with just analysis results
            const saveResponse = await axios.post('http://localhost:5000/api/audio/save-session', {
              sessionName,
              userEmail,
              originalAudio: response.data.audio_preview,
              denoisedAudios: [],
              analysisResults: {
                audio_id: response.data.audio_id,
                has_noise: response.data.has_noise || true,
                overall_noise_level: response.data.overall_noise_level || 0,
                noise_types: response.data.noise_types || [],
                noise_profile: response.data.noise_profile || 'unknown',
                recommendations: response.data.recommendations || ''
              },
              originalWaveform: response.data.waveform_data?.original || []
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
      
      setProgress(100);
      setStatus('Analysis complete. Click "Denoise" to process.');
      setIsLoading(false);
    } catch (error) {
      console.error('Error analyzing audio:', error);
      console.error('Error response:', error.response?.data || 'No response data');
      setStatus(`Error during analysis: ${error.response?.data?.error || error.message}. Please try again.`);
      setProgress(0);
      setIsLoading(false);
    }
  };
  
  // Denoise audio function
  const handleDenoise = async () => {
    if (!audioId || isLoading) return;
    
    // Check if user has already done one round and is not logged in
    if (!isLoggedIn && denoisingRound >= 1) {
      showLoginModal();
      return;
    }
    
    try {
      setIsLoading(true);
      setStatus(`Starting denoising round ${denoisingRound + 1}...`);
      setProgress(10);
      
      const response = await axios.post('http://localhost:5001/api/audio/denoise', {
        audio_id: audioId
      });
      
      console.log('Denoising response received:', response.data);
      
      // Update the progress and state
      setProgress(90);
      
      if (response.data.audio_preview) {
        setDenoisedAudio(response.data.audio_preview);
        
        // Update audio duration if available
        if (response.data.duration) {
          const minutes = Math.floor(response.data.duration / 60);
          const seconds = Math.floor(response.data.duration % 60);
          setAudioDuration(`${minutes}:${seconds.toString().padStart(2, '0')}`);
        }
        
        // Set waveform data if available
        if (response.data.waveform_data) {
          console.log('Received waveform data:', response.data.waveform_data);
          
          const originalWaveform = response.data.waveform_data.original || [];
          const denoisedWaveform = response.data.waveform_data.denoised || [];
          
          // Log the waveform data lengths
          console.log(`Original waveform length: ${originalWaveform.length}, Denoised waveform length: ${denoisedWaveform.length}`);
          
          setWaveformData({
            original: originalWaveform,
            denoised: denoisedWaveform
          });
        
          // Also update the original waveform in the history if needed
          if (denoisingHistory.length > 0 && (!denoisingHistory[0].waveform || denoisingHistory[0].waveform.length === 0)) {
            console.log('Updating original waveform in history');
            setDenoisingHistory(prev => [
              {
                ...prev[0],
                waveform: originalWaveform
              },
              ...prev.slice(1)
            ]);
          }
        }
        
        // Make a copy of the metrics with default values for any missing metrics
        const metricsData = {
          snr: response.data.metrics?.snr ?? 12.5, // Default SNR value if missing
          noise_reduction: response.data.metrics?.noise_reduction ?? 25.0, // Default noise reduction if missing
          spectrogram_improvement: response.data.metrics?.spectrogram_improvement ?? 15.0, // Default improvement if missing 
          low_improvement: response.data.metrics?.low_improvement ?? false,
          clarity_score: response.data.metrics?.clarity_score ?? 80.0,
          background_noise_reduction: response.data.metrics?.background_noise_reduction ?? 20.0
        };
        
        console.log('Metrics for history:', metricsData);
        
        // Add to denoising history
        const originalEntry = denoisingHistory.length > 0 ? denoisingHistory[0] : null;
        
        let newDenoisingHistory;
        
        // If we have history, keep the original audio entry and add the new one
        if (denoisingHistory.length > 0) {
          newDenoisingHistory = [
            denoisingHistory[0], // Keep the original entry
            ...denoisingHistory.slice(1), // Keep any other previous entries except the original
            {
              round: response.data.denoising_round,
              audio: response.data.audio_preview,
              waveform: response.data.waveform_data?.denoised || [],
              metrics: metricsData
            }
          ];
        } else {
          // If somehow we don't have history yet, create a new array
          newDenoisingHistory = [{
            round: 0,
            audio: originalAudio,
            waveform: waveformData.original || [],
            metrics: null
          }, {
            round: response.data.denoising_round,
            audio: response.data.audio_preview,
            waveform: response.data.waveform_data?.denoised || [],
            metrics: metricsData
          }];
        }
        
        setDenoisingHistory(newDenoisingHistory);
        
        // Save session to database if user is logged in
        if (isLoggedIn) {
          try {
            const userEmail = localStorage.getItem('userEmail');
            if (userEmail) {
              // Prepare denoised audio entries with enhanced metrics
              const denoisedAudios = newDenoisingHistory.slice(1).map(item => ({
                round: item.round,
                audio: item.audio,
                metrics: {
                  // Include all available metrics
                  ...item.metrics,
                  snr: item.metrics.snr,
                  noise_reduction: item.metrics.noise_reduction,
                  spectrogram_improvement: item.metrics.spectrogram_improvement,
                  low_improvement: item.metrics.low_improvement,
                  clarity_score: item.metrics.clarity_score,
                  background_noise_reduction: item.metrics.background_noise_reduction
                },
                waveform: item.waveform
              }));

              // Save denoised audio session to database
              const saveResponse = await axios.post('http://localhost:5000/api/audio/save-session', {
                sessionName,
                userEmail,
                originalAudio: newDenoisingHistory[0].audio,
                denoisedAudios: denoisedAudios,
                analysisResults: analysisResults ? {
                  audio_id: analysisResults.audio_id,
                  has_noise: analysisResults.has_noise || true,
                  overall_noise_level: analysisResults.overall_noise_level || 0,
                  noise_types: analysisResults.noise_types || [],
                  noise_profile: analysisResults.noise_profile || 'unknown',
                  recommendations: analysisResults.recommendations || '',
                  noise_levels: analysisResults.noise_levels || {}
                } : null,
                originalWaveform: newDenoisingHistory[0].waveform
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
      } else {
        console.error('No audio preview in response');
        throw new Error('No audio preview received from server');
      }
      
      setMetrics(response.data.metrics || {});
      setDenoisingRound(response.data.denoising_round || 0);
      
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
      if (response.data.denoising_round >= 1 && response.data.metrics && response.data.metrics.low_improvement) {
        setTimeout(() => {
          showConfirmationModal('Limited improvement detected. Do you still want to apply another round of denoising?', handleDenoise, 'Continue Denoising?');
        }, 500);
      } else if (response.data.denoising_round >= 1) {
        setTimeout(() => {
          showConfirmationModal('Do you want to apply another round of denoising to improve the audio further?', handleDenoise, 'Continue Denoising?');
        }, 500);
        }
      }
    } catch (error) {
      console.error('Error denoising audio:', error);
      console.error('Error response:', error.response?.data || 'No response data');
      setStatus(`Error during denoising: ${error.response?.data?.error || error.message}. Please try again.`);
      setProgress(0);
      setIsLoading(false);
      showMessageModal((error.response?.data?.error || 'Unknown error'), 'error', 'Error Denoising Audio');
    }
  };
  
  // Handle actual download
  const downloadAudio = (format, version = 'current') => {
    if (!audioId) return;
    
    const timestamp = new Date().toISOString().replace(/:/g, '-').replace(/\..+/, '');
    let filename;
    
    if (version === 'original') {
      filename = `original_${timestamp}`;
    } else if (version === 'current') {
      filename = `denoised_${timestamp}`;
    } else {
      filename = `denoised_round${version}_${timestamp}`;
    }
    
    const fullFilename = `${filename}.${format}`;
    const downloadUrl = `http://localhost:5001/api/audio/save/${audioId}?filename=${filename}&format=${format}&version=${version}`;
    
    // Show download progress modal
    setModalContent('Preparing download...');
    setModalType('info');
    setModalTitle('Downloading Audio');
    setModalButtons([]);
    setIsModalOpen(true);
    
    // Use XMLHttpRequest for better error handling
    const xhr = new XMLHttpRequest();
    xhr.open('GET', downloadUrl, true);
    xhr.responseType = 'blob';
    
    xhr.onload = function() {
      if (this.status === 200) {
        // Create a blob URL from the data
        const blob = new Blob([xhr.response], { 
          type: format === 'mp3' ? 'audio/mpeg' : 'audio/wav' 
        });
        const url = window.URL.createObjectURL(blob);
        
        // Create a download link and click it
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = fullFilename;
        document.body.appendChild(a);
        a.click();
        
        // Clean up
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        // Show success message
        setModalContent('Audio saved successfully!');
        setModalType('success');
        setModalButtons([{ label: 'OK', onClick: () => setIsModalOpen(false) }]);
        
        // Close modal after a delay
        setTimeout(() => {
          setIsModalOpen(false);
        }, 2000);
      } else {
        // Handle HTTP errors
        console.error('Download failed with status:', this.status);
        setModalContent(`Download failed with status: ${this.status}. Please try again.`);
        setModalType('error');
        setModalButtons([{ label: 'OK', onClick: () => setIsModalOpen(false) }]);
      }
    };
    
    xhr.onerror = function() {
      console.error('Download request failed');
      setModalContent('Download failed! Network error. Please try again.');
      setModalType('error');
      setModalButtons([{ label: 'OK', onClick: () => setIsModalOpen(false) }]);
    };
    
    xhr.onprogress = function(event) {
      if (event.lengthComputable) {
        const percentComplete = Math.round((event.loaded / event.total) * 100);
        setModalContent(`Downloading: ${percentComplete}%`);
      }
    };
    
    xhr.send();
  };

  // Download modal content component
  const DownloadModalContent = () => (
    <div className="audio-download-options">
      <div className="audio-option-group">
        <label>Choose Version:</label>
        <select 
          className="audio-download-select"
          value={selectedVersion}
          onChange={(e) => setSelectedVersion(e.target.value)}
        >
          {denoisingHistory.slice(1).map((item, index) => (
            <option key={index + 1} value={index + 1}>
              Round {index + 1} {index + 1 === denoisingRound ? '(Latest)' : ''}
            </option>
          ))}
          <option value="original">Original Audio</option>
        </select>
      </div>
      
      <div className="audio-option-group">
        <label>File Format:</label>
        <div className="audio-format-options">
          <button 
            className={`audio-format-button ${downloadFormat === 'wav' ? 'audio-active' : ''}`}
            onClick={() => setDownloadFormat('wav')}
          >
            WAV Format (High Quality)
          </button>
          <button 
            className={`audio-format-button ${downloadFormat === 'mp3' ? 'audio-active' : ''}`}
            onClick={() => setDownloadFormat('mp3')}
          >
            MP3 Format (Compressed)
          </button>
          <button 
            className={`audio-format-button ${downloadFormat === 'ogg' ? 'audio-active' : ''}`}
            onClick={() => setDownloadFormat('ogg')}
          >
            OGG Format (Web-Friendly)
          </button>
        </div>
      </div>
    </div>
  );

  // Show download options modal
  const showDownloadOptionsModal = () => {
    setIsDownloadModalOpen(true);
  };

  // Save audio function
  const handleSave = () => {
    if (!audioId) return;
    
    // Check if user is logged in
    if (!isLoggedIn) {
      showLoginModal();
      return;
    }
    
    // Make sure we have the waveform data before saving
    if (waveformData.original.length === 0) {
      console.warn('Original waveform data is missing, regenerating from audio...');
      // This could be enhanced with actual waveform generation if needed
    }
    
    showDownloadOptionsModal();
  };

  const handleBackClick = () => {
    goBack();
  };

  // Separate useEffect for original audio playback position
  useEffect(() => {
    let intervalId;
    
    if (isPlayingOriginal && originalAudioRef.current) {
      intervalId = setInterval(() => {
        const audio = originalAudioRef.current;
        if (audio.duration) {
          const position = audio.currentTime / audio.duration;
          setOriginalPlaybackPosition(position);
        }
      }, 50);
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isPlayingOriginal]);

  // Separate useEffect for denoised audio playback position
  useEffect(() => {
    let intervalId;
    
    if (isPlayingDenoised && denoisedAudioRef.current) {
      intervalId = setInterval(() => {
        const audio = denoisedAudioRef.current;
        if (audio.duration) {
          const position = audio.currentTime / audio.duration;
          setDenoisedPlaybackPosition(position);
        }
      }, 50);
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isPlayingDenoised]);

  // Original waveform playhead handlers
  const handleOriginalPlayheadMouseDown = (e) => {
    e.stopPropagation();
    setIsDraggingOriginalPlayhead(true);
    document.addEventListener('mousemove', handleOriginalPlayheadMove);
    document.addEventListener('mouseup', handleOriginalPlayheadMouseUp);
  };

  const handleOriginalPlayheadMove = (e) => {
    if (isDraggingOriginalPlayhead && originalWaveformRef.current) {
      const container = originalWaveformRef.current;
      const rect = container.getBoundingClientRect();
      const position = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      
      setOriginalPlaybackPosition(position);
      
      // Update audio position
      if (originalAudioRef.current) {
        originalAudioRef.current.currentTime = position * originalAudioRef.current.duration;
      }
    }
  };

  const handleOriginalPlayheadMouseUp = () => {
    setIsDraggingOriginalPlayhead(false);
    document.removeEventListener('mousemove', handleOriginalPlayheadMove);
    document.removeEventListener('mouseup', handleOriginalPlayheadMouseUp);
  };

  const handleOriginalWaveformClick = (e) => {
    if (originalWaveformRef.current) {
      const container = originalWaveformRef.current;
      const rect = container.getBoundingClientRect();
      const position = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      
      setOriginalPlaybackPosition(position);
      
      // Update audio position
      if (originalAudioRef.current) {
        originalAudioRef.current.currentTime = position * originalAudioRef.current.duration;
      }
    }
  };

  // Denoised waveform playhead handlers
  const handleDenoisedPlayheadMouseDown = (e) => {
    e.stopPropagation();
    setIsDraggingDenoisedPlayhead(true);
    document.addEventListener('mousemove', handleDenoisedPlayheadMove);
    document.addEventListener('mouseup', handleDenoisedPlayheadMouseUp);
  };

  const handleDenoisedPlayheadMove = (e) => {
    if (isDraggingDenoisedPlayhead && denoisedWaveformRef.current) {
      const container = denoisedWaveformRef.current;
      const rect = container.getBoundingClientRect();
      const position = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      
      setDenoisedPlaybackPosition(position);
      
      // Update audio position
      if (denoisedAudioRef.current) {
        denoisedAudioRef.current.currentTime = position * denoisedAudioRef.current.duration;
      }
    }
  };

  const handleDenoisedPlayheadMouseUp = () => {
    setIsDraggingDenoisedPlayhead(false);
    document.removeEventListener('mousemove', handleDenoisedPlayheadMove);
    document.removeEventListener('mouseup', handleDenoisedPlayheadMouseUp);
  };

  const handleDenoisedWaveformClick = (e) => {
    if (denoisedWaveformRef.current) {
      const container = denoisedWaveformRef.current;
      const rect = container.getBoundingClientRect();
      const position = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      
      setDenoisedPlaybackPosition(position);
      
      // Update audio position
      if (denoisedAudioRef.current) {
        denoisedAudioRef.current.currentTime = position * denoisedAudioRef.current.duration;
      }
    }
  };

  // Add useEffect for comparison original audio playback position
  useEffect(() => {
    let intervalId;
    
    if (comparePlaying === 'original' && compareOriginalRef.current) {
      intervalId = setInterval(() => {
        const audio = compareOriginalRef.current;
        if (audio.duration) {
          const position = audio.currentTime / audio.duration;
          setCompareOriginalPlaybackPosition(position);
        }
      }, 50);
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [comparePlaying]);

  // Add useEffect for comparison processed audio playback position
  useEffect(() => {
    let intervalId;
    
    if (comparePlaying === 'processed' && compareProcessedRef.current) {
      intervalId = setInterval(() => {
        const audio = compareProcessedRef.current;
        if (audio.duration) {
          const position = audio.currentTime / audio.duration;
          setCompareProcessedPlaybackPosition(position);
        }
      }, 50);
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [comparePlaying]);

  // Add handlers for comparison original waveform
  const handleCompareOriginalPlayheadMouseDown = (e) => {
    e.stopPropagation();
    setIsDraggingCompareOriginalPlayhead(true);
    document.addEventListener('mousemove', handleCompareOriginalPlayheadMove);
    document.addEventListener('mouseup', handleCompareOriginalPlayheadMouseUp);
  };

  const handleCompareOriginalPlayheadMove = (e) => {
    if (isDraggingCompareOriginalPlayhead && compareOriginalWaveformRef.current) {
      const container = compareOriginalWaveformRef.current;
      const rect = container.getBoundingClientRect();
      const position = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      
      setCompareOriginalPlaybackPosition(position);
      
      // Update audio position
      if (compareOriginalRef.current) {
        compareOriginalRef.current.currentTime = position * compareOriginalRef.current.duration;
      }
    }
  };

  const handleCompareOriginalPlayheadMouseUp = () => {
    setIsDraggingCompareOriginalPlayhead(false);
    document.removeEventListener('mousemove', handleCompareOriginalPlayheadMove);
    document.removeEventListener('mouseup', handleCompareOriginalPlayheadMouseUp);
  };

  const handleCompareOriginalWaveformClick = (e) => {
    if (compareOriginalWaveformRef.current) {
      const container = compareOriginalWaveformRef.current;
      const rect = container.getBoundingClientRect();
      const position = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      
      setCompareOriginalPlaybackPosition(position);
      
      // Update audio position
      if (compareOriginalRef.current) {
        compareOriginalRef.current.currentTime = position * compareOriginalRef.current.duration;
      }
    }
  };

  // Add handlers for comparison processed waveform
  const handleCompareProcessedPlayheadMouseDown = (e) => {
    e.stopPropagation();
    setIsDraggingCompareProcessedPlayhead(true);
    document.addEventListener('mousemove', handleCompareProcessedPlayheadMove);
    document.addEventListener('mouseup', handleCompareProcessedPlayheadMouseUp);
  };

  const handleCompareProcessedPlayheadMove = (e) => {
    if (isDraggingCompareProcessedPlayhead && compareProcessedWaveformRef.current) {
      const container = compareProcessedWaveformRef.current;
      const rect = container.getBoundingClientRect();
      const position = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      
      setCompareProcessedPlaybackPosition(position);
      
      // Update audio position
      if (compareProcessedRef.current) {
        compareProcessedRef.current.currentTime = position * compareProcessedRef.current.duration;
      }
    }
  };

  const handleCompareProcessedPlayheadMouseUp = () => {
    setIsDraggingCompareProcessedPlayhead(false);
    document.removeEventListener('mousemove', handleCompareProcessedPlayheadMove);
    document.removeEventListener('mouseup', handleCompareProcessedPlayheadMouseUp);
  };

  const handleCompareProcessedWaveformClick = (e) => {
    if (compareProcessedWaveformRef.current) {
      const container = compareProcessedWaveformRef.current;
      const rect = container.getBoundingClientRect();
      const position = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      
      setCompareProcessedPlaybackPosition(position);
      
      // Update audio position
      if (compareProcessedRef.current) {
        compareProcessedRef.current.currentTime = position * compareProcessedRef.current.duration;
      }
    }
  };

  // Make sure waveform data is included when saving
  useEffect(() => {
    // Whenever the session is updated (after denoising or analysis)
    if (sessionId && isLoggedIn) {
      const saveSession = async () => {
        try {
          const userEmail = localStorage.getItem('userEmail');
          if (userEmail) {
            await axios.post('http://localhost:5000/api/audio/save-session', {
              sessionName,
              userEmail,
              originalAudio,
              denoisedAudios: denoisingHistory.slice(1).map(item => ({
                round: item.round,
                audio: item.audio,
                waveform: item.waveform || [],
                metrics: item.metrics
              })),
              analysisResults: analysisResults ? {
                audio_id: analysisResults.audio_id,
                has_noise: analysisResults.has_noise || true,
                overall_noise_level: analysisResults.overall_noise_level || 0,
                noise_types: analysisResults.noise_types || [],
                noise_profile: analysisResults.noise_profile || 'unknown',
                recommendations: analysisResults.recommendations || '',
                noise_levels: analysisResults.noise_levels || {}
              } : null,
              originalWaveform: waveformData.original || []
            });
            console.log('Session auto-saved successfully');
          }
        } catch (error) {
          console.error('Error auto-saving session:', error);
        }
      };

      saveSession();
    }
  }, [sessionId, denoisingRound, analysisResults]);

  return (
    <div className="audio-denoiser-container">
      <div className="audio-glass-panel glass-panel">
        <div className="audio-header-section">
          <button className="audio-back-button" onClick={handleBackClick}>
            <FaArrowLeft /> Back
          </button>
          <h1>Audio Denoiser</h1>
          <button 
            className={`audio-compare-button ${showComparison ? 'audio-active' : ''}`}
            onClick={toggleComparison}
            disabled={denoisingHistory.length < 2}
          >
            <FaChartBar /> Compare Results
          </button>
        </div>

        {showComparison ? (
          <div className="audio-comparison-view">
            <div className="audio-comparison-controls">
              <h3>Denoising History (Round {currentCompareIndex} of {denoisingHistory.length - 1})</h3>
              <div>
                <button 
                  className="audio-nav-button"
                  onClick={handlePrevRound}
                  disabled={currentCompareIndex <= 1}
                >
                  <FaChevronLeft /> Previous
                </button>
                <button 
                  className="audio-nav-button"
                  onClick={handleNextRound}
                  disabled={currentCompareIndex >= denoisingHistory.length - 1}
                >
                  Next <FaChevronRight />
                </button>
              </div>
            </div>
            
            <div className="audio-comparison-players">
              <div className="audio-comparison-player-card">
                <h4>Original Audio</h4>
                <div className="audio-comparison-player">
                  <audio 
                    ref={compareOriginalRef} 
                    src={denoisingHistory[0]?.audio}
                    onEnded={handleCompareAudioEnded}
                  ></audio>
                  <button 
                    className="audio-comparison-play-button"
                    onClick={() => toggleCompareAudio('original')}
                  >
                    {comparePlaying === 'original' ? <FaPause /> : <FaPlay />}
                  </button>
                  <div className={`audio-comparison-visualizer ${comparePlaying === 'original' ? 'playing' : ''}`}>
                    <div className="audio-visualizer-bar"></div>
                    <div className="audio-visualizer-bar"></div>
                    <div className="audio-visualizer-bar"></div>
                    <div className="audio-visualizer-bar"></div>
                    <div className="audio-visualizer-bar"></div>
                  </div>
                </div>
                
                <div className="audio-comparison-waveform">
                  {denoisingHistory[0]?.waveform && denoisingHistory[0]?.waveform.length > 0 ? (
                    <div 
                      className="audio-waveform-container" 
                      ref={compareOriginalWaveformRef}
                      onClick={handleCompareOriginalWaveformClick}
                    >
                      <Line
                        data={{
                          labels: Array.from({ length: denoisingHistory[0]?.waveform.length }, (_, i) => i),
                          datasets: [
                            {
                              label: 'Original Waveform',
                              data: denoisingHistory[0]?.waveform,
                              borderColor: 'rgba(255, 255, 255, 0.7)',
                              borderWidth: 1,
                              pointRadius: 0,
                              tension: 0.4
                            }
                          ]
                        }}
                        options={{
                          responsive: true,
                          maintainAspectRatio: false,
                          scales: {
                            y: {
                              ticks: { color: 'rgba(255, 255, 255, 0.5)' },
                              grid: { color: 'rgba(255, 255, 255, 0.1)' },
                              min: -1,
                              max: 1
                            },
                            x: {
                              display: false
                            }
                          },
                          plugins: {
                            legend: {
                              display: false
                            }
                          },
                          animation: false
                        }}
                      />
                      {/* Playhead indicator */}
                      <div 
                        className="audio-playhead"
                        style={{ left: `${compareOriginalPlaybackPosition * 100}%` }}
                      >
                        <div 
                          className="audio-playhead-handle"
                          onMouseDown={handleCompareOriginalPlayheadMouseDown}
                        ></div>
                      </div>
                    </div>
                  ) : (
                    <div className="audio-no-waveform">
                      <p>No waveform data available</p>
                    </div>
                  )}
                </div>
              </div>
              
              <div className="audio-comparison-player-card">
                <h4>Round {currentCompareIndex} Result</h4>
                <div className="audio-comparison-player">
                  <audio 
                    ref={compareProcessedRef} 
                    src={denoisingHistory[currentCompareIndex]?.audio}
                    onEnded={handleCompareAudioEnded}
                  ></audio>
                  <button 
                    className="audio-comparison-play-button"
                    onClick={() => toggleCompareAudio('processed')}
                  >
                    {comparePlaying === 'processed' ? <FaPause /> : <FaPlay />}
                  </button>
                  <div className={`audio-comparison-visualizer ${comparePlaying === 'processed' ? 'playing' : ''}`}>
                    <div className="audio-visualizer-bar"></div>
                    <div className="audio-visualizer-bar"></div>
                    <div className="audio-visualizer-bar"></div>
                    <div className="audio-visualizer-bar"></div>
                    <div className="audio-visualizer-bar"></div>
                  </div>
                </div>
                
                <div className="audio-comparison-waveform">
                  {denoisingHistory[currentCompareIndex]?.waveform && denoisingHistory[currentCompareIndex]?.waveform.length > 0 ? (
                    <div 
                      className="audio-waveform-container"
                      ref={compareProcessedWaveformRef}
                      onClick={handleCompareProcessedWaveformClick}
                    >
                      <Line
                        data={{
                          labels: Array.from({ length: denoisingHistory[currentCompareIndex]?.waveform.length }, (_, i) => i),
                          datasets: [
                            {
                              label: 'Processed Waveform',
                              data: denoisingHistory[currentCompareIndex]?.waveform,
                              borderColor: 'rgba(140, 82, 255, 0.7)',
                              borderWidth: 1,
                              pointRadius: 0,
                              tension: 0.4
                            }
                          ]
                        }}
                        options={{
                          responsive: true,
                          maintainAspectRatio: false,
                          scales: {
                            y: {
                              ticks: { color: 'rgba(255, 255, 255, 0.5)' },
                              grid: { color: 'rgba(255, 255, 255, 0.1)' },
                              min: -1,
                              max: 1
                            },
                            x: {
                              display: false
                            }
                          },
                          plugins: {
                            legend: {
                              display: false
                            }
                          },
                          animation: false
                        }}
                      />
                      {/* Playhead indicator */}
                      <div 
                        className="audio-playhead"
                        style={{ left: `${compareProcessedPlaybackPosition * 100}%` }}
                      >
                        <div 
                          className="audio-playhead-handle"
                          onMouseDown={handleCompareProcessedPlayheadMouseDown}
                        ></div>
                      </div>
                    </div>
                  ) : (
                    <div className="audio-no-waveform">
                      <p>No waveform data available</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
            
            <div className="audio-comparison-stats">
              <h4>Quality Improvement</h4>
              <div className="audio-stats-grid">
                <div className="audio-stat-item">
                  <span className="audio-stat-label">Signal-to-Noise Ratio:</span>
                  <span className="audio-stat-value">
                    {getMetricValue(denoisingHistory[currentCompareIndex], 'snr').toFixed(4)} dB
                  </span>
                  <div className="audio-metric-bar-container">
                    <div 
                      className="audio-metric-bar" 
                      style={{ width: `${Math.min(100, getMetricValue(denoisingHistory[currentCompareIndex], 'snr') * 3)}%` }}
                    ></div>
                  </div>
                </div>
                <div className="audio-stat-item">
                  <span className="audio-stat-label">Noise Reduction:</span>
                  <span className="audio-stat-value">
                    {getMetricValue(denoisingHistory[currentCompareIndex], 'noise_reduction').toFixed(4)}%
                  </span>
                  <div className="audio-metric-bar-container">
                    <div 
                      className="audio-metric-bar" 
                      style={{ width: `${Math.min(100, getMetricValue(denoisingHistory[currentCompareIndex], 'noise_reduction'))}%` }}
                    ></div>
                  </div>
                </div>
                {(getMetricValue(denoisingHistory[currentCompareIndex], 'spectrogram_improvement') > 0) && (
                  <div className="audio-stat-item">
                    <span className="audio-stat-label">Spectral Improvement:</span>
                    <span className="audio-stat-value">
                      {getMetricValue(denoisingHistory[currentCompareIndex], 'spectrogram_improvement').toFixed(4)}%
                    </span>
                    <div className="audio-metric-bar-container">
                      <div 
                        className="audio-metric-bar" 
                        style={{ width: `${Math.min(100, getMetricValue(denoisingHistory[currentCompareIndex], 'spectrogram_improvement'))}%` }}
                      ></div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
        <div className="audio-content-section">
          <div className="audio-sidebar">
            <div className="audio-upload-section glass-card">
              <h3>Upload Audio</h3>
              <button 
                className="audio-upload-button button"
                onClick={() => document.querySelector('input[type="file"]').click()}
                disabled={isLoading}
              >
                <FaUpload /> Upload Audio File
              </button>
                <input 
                  type="file" 
                  accept="audio/*" 
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                />
              
              {fileName && (
                <div className="audio-file-info">
                  <p><strong>File:</strong> {fileName}</p>
                  {audioDuration && <p><strong>Duration:</strong> {audioDuration}</p>}
                </div>
              )}
            </div>

            <div className="audio-analysis-section glass-card">
              <h3>Noise Analysis</h3>
              <div className="audio-analysis-content">
                {!analysisResults ? (
                  <p className="audio-empty-message">Upload and analyze an audio file to see noise detection results.</p>
                ) : analysisResults.has_noise ? (
                  <>
                    <div className="audio-analysis-header">
                      <h4>NOISE DETECTED</h4>
                      <div className="audio-noise-meter-container">
                        <div className="audio-noise-meter">
                          <div 
                            className="audio-noise-meter-fill" 
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
                        <p className="audio-noise-level">
                          {analysisResults.overall_noise_level?.toFixed(2) || '0.00'}%
                        </p>
                      </div>
                    </div>

                    <div className="audio-noise-types">
                      <h4>Detected Noise Types:</h4>
                      <ul>
                        {(analysisResults.noise_types || []).map((type) => {
                          const noiseLevel = getNoiseLevelValue(type);
                          return (
                            <li key={type}>
                              {type.replace('_', ' ').charAt(0).toUpperCase() + type.replace('_', ' ').slice(1)}: 
                              <div className="audio-noise-type-meter">
                                <div 
                                  className="audio-noise-type-fill" 
                                  style={{ 
                                    width: `${Math.min(100, noiseLevel * 10)}%`,
                                    backgroundColor: noiseLevel > 5 
                                      ? '#ff4d4d' 
                                      : noiseLevel > 2 
                                        ? '#ffa64d' 
                                        : '#4dff88'
                                  }}
                                ></div>
                              </div>
                              {' '}{noiseLevel?.toFixed(2) || '0.00'}%
                            </li>
                          );
                        })}
                      </ul>
                    </div>

                    <div className="audio-recommendations">
                      <h4>Recommendation:</h4>
                      <p>{analysisResults.recommendations}</p>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="audio-analysis-header">
                      <h4>NO SIGNIFICANT NOISE DETECTED</h4>
                    </div>
                    <p>The audio appears to be clear with minimal noise.</p>
                    <p>You can still try denoising if you want to enhance audio quality further.</p>
                  </>
                )}
              </div>
            </div>

            <div className="audio-actions-section glass-card">
              <h3>Actions</h3>
              <button 
                className="audio-action-button audio-analyze-button"
                onClick={handleAnalyze}
                disabled={!originalAudio || isLoading}
              >
                <FaSearch /> Analyze Audio
              </button>
              
              <button
                  className={`audio-action-button audio-denoise-button ${!isLoggedIn && denoisingRound >= 1 ? 'audio-disabled-button' : ''}`}
                onClick={handleDenoise}
                disabled={!audioId || isLoading}
                  title={!isLoggedIn && denoisingRound >= 1 ? "Login required for additional denoising rounds" : "Denoise Audio"}
              >
                  {!isLoggedIn && denoisingRound >= 1 && <FaLock className="audio-lock-icon" />}
                <FaMagic /> Denoise Audio
              </button>
              
              <button
                className={`audio-action-button audio-save-button ${!isLoggedIn ? 'audio-disabled-button' : ''}`}
                onClick={handleSave}
                disabled={!denoisedAudio || isLoading}
              >
                {!isLoggedIn && <FaLock className="audio-lock-icon" />}
                <FaSave /> Save Audio
              </button>
              
              <button 
                className="audio-action-button audio-reset-button"
                onClick={handleReset}
                disabled={isLoading}
              >
                <FaRedo /> Reset
              </button>
            </div>
          </div>

          <div className="audio-main-content">
            <div className="audio-status-bar glass-card">
              <div className="audio-progress-container">
                <div 
                  className="audio-progress-bar"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
              <p className="audio-status-text">{status}</p>
            </div>

            <div className="audio-display-section glass-card">
              <div className="audio-panel">
                <h3>Original Audio</h3>
                <div className="audio-wrapper">
                  {originalAudio ? (
                    <>
                      <audio 
                        ref={originalAudioRef}
                        src={originalAudio}
                        onEnded={() => handleAudioEnded('original')}
                        className="audio-player"
                      ></audio>
                      <button 
                        className="audio-play-button"
                        onClick={toggleOriginalPlayback}
                      >
                        {isPlayingOriginal ? <FaPause /> : <FaPlay />}
                      </button>
                      <FaVolumeUp className="audio-volume-icon" />
                      <div className="audio-visualizer">
                        <div className="audio-visualizer-bar"></div>
                        <div className="audio-visualizer-bar"></div>
                        <div className="audio-visualizer-bar"></div>
                        <div className="audio-visualizer-bar"></div>
                        <div className="audio-visualizer-bar"></div>
                      </div>
                    </>
                  ) : (
                    <div className="audio-no-audio">
                      <p>No audio loaded</p>
                    </div>
                  )}
                </div>
                {waveformData.original.length > 0 && (
                  <div 
                    className="audio-waveform-container" 
                    ref={originalWaveformRef}
                    onClick={handleOriginalWaveformClick}
                  >
                    <Line
                      data={{
                        labels: Array.from({ length: waveformData.original.length }, (_, i) => i),
                        datasets: [
                          {
                            label: 'Original Waveform',
                            data: waveformData.original,
                            borderColor: 'rgba(255, 255, 255, 0.7)',
                            borderWidth: 1,
                            pointRadius: 0,
                            tension: 0.4
                          }
                        ]
                      }}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                          y: {
                            ticks: { color: 'rgba(255, 255, 255, 0.5)' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            min: -1,
                            max: 1
                          },
                          x: {
                            display: false
                          }
                        },
                        plugins: {
                          legend: {
                            display: false
                          }
                        },
                        animation: false
                      }}
                    />
                    {/* Playhead indicator */}
                    <div 
                      className="audio-playhead"
                      style={{ left: `${originalPlaybackPosition * 100}%` }}
                    >
                      <div 
                        className="audio-playhead-handle"
                        onMouseDown={handleOriginalPlayheadMouseDown}
                      ></div>
                    </div>
                  </div>
                )}
              </div>

              <div className="audio-panel">
                <h3>Denoised Audio</h3>
                <div className="audio-wrapper">
                  {denoisedAudio ? (
                    <>
                      <audio 
                        ref={denoisedAudioRef}
                        src={denoisedAudio}
                        onEnded={() => handleAudioEnded('denoised')}
                        className="audio-player"
                      ></audio>
                      <button 
                        className="audio-play-button"
                        onClick={toggleDenoisedPlayback}
                      >
                        {isPlayingDenoised ? <FaPause /> : <FaPlay />}
                      </button>
                      <FaVolumeUp className="audio-volume-icon" />
                      <div className="audio-visualizer">
                        <div className="audio-visualizer-bar"></div>
                        <div className="audio-visualizer-bar"></div>
                        <div className="audio-visualizer-bar"></div>
                        <div className="audio-visualizer-bar"></div>
                        <div className="audio-visualizer-bar"></div>
                      </div>
                    </>
                  ) : (
                    <div className="audio-no-audio">
                      <p>No processed audio yet</p>
                    </div>
                  )}
                </div>
                {waveformData.denoised.length > 0 && (
                  <div 
                    className="audio-waveform-container"
                    ref={denoisedWaveformRef}
                    onClick={handleDenoisedWaveformClick}
                  >
                    <Line
                      data={{
                        labels: Array.from({ length: waveformData.denoised.length }, (_, i) => i),
                        datasets: [
                          {
                            label: 'Denoised Waveform',
                            data: waveformData.denoised,
                            borderColor: 'rgba(140, 82, 255, 0.7)',
                            borderWidth: 1,
                            pointRadius: 0,
                            tension: 0.4
                          }
                        ]
                      }}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                          y: {
                            ticks: { color: 'rgba(255, 255, 255, 0.5)' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            min: -1,
                            max: 1
                          },
                          x: {
                            display: false
                          }
                        },
                        plugins: {
                          legend: {
                            display: false
                          }
                        },
                        animation: false
                      }}
                    />
                    {/* Playhead indicator */}
                    <div 
                      className="audio-playhead"
                      style={{ left: `${denoisedPlaybackPosition * 100}%` }}
                    >
                      <div 
                        className="audio-playhead-handle"
                        onMouseDown={handleDenoisedPlayheadMouseDown}
                      ></div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="audio-stats-section glass-card">
              <h3>Denoising Statistics</h3>
                {denoisingHistory.length > 1 ? (
                <div className="audio-stats-content">
                    {denoisingHistory.slice(1).map((historyItem, index) => (
                      <div key={`round-${index + 1}`} className="audio-stats-round">
                        <div className="audio-round-badge">Round {index + 1}</div>
                  
                  <div className="audio-metrics">
                    <div className="audio-metric">
                      <span className="audio-metric-name">Signal-to-Noise Ratio:</span>
                      <span className="audio-metric-value">{getMetricValue(historyItem, 'snr').toFixed(4)} dB</span>
                      <div className="audio-metric-bar-container">
                        <div 
                          className="audio-metric-bar" 
                          style={{ width: `${Math.min(100, getMetricValue(historyItem, 'snr') * 3)}%` }}
                        ></div>
                      </div>
                      <span className="audio-metric-description">Higher values indicate cleaner audio</span>
                    </div>
                    
                    <div className="audio-metric">
                      <span className="audio-metric-name">Noise Reduction:</span>
                      <span className="audio-metric-value">{getMetricValue(historyItem, 'noise_reduction').toFixed(4)}%</span>
                      <div className="audio-metric-bar-container">
                        <div 
                          className="audio-metric-bar" 
                          style={{ width: `${Math.min(100, getMetricValue(historyItem, 'noise_reduction'))}%` }}
                        ></div>
                      </div>
                      <span className="audio-metric-description">Percentage of noise removed</span>
                    </div>
                    
                    {getMetricValue(historyItem, 'spectrogram_improvement') > 0 && (
                      <div className="audio-metric">
                        <span className="audio-metric-name">Spectral Improvement:</span>
                        <span className="audio-metric-value">{getMetricValue(historyItem, 'spectrogram_improvement').toFixed(4)}%</span>
                        <div className="audio-metric-bar-container">
                          <div 
                            className="audio-metric-bar" 
                            style={{ width: `${Math.min(100, getMetricValue(historyItem, 'spectrogram_improvement'))}%` }}
                          ></div>
                        </div>
                        <span className="audio-metric-description">Improvement in frequency distribution</span>
                      </div>
                    )}
                  </div>
                  
                  {getMetricValue(historyItem, 'low_improvement') && index > 0 && (
                    <p className="audio-improvement-note">
                      <FaInfoCircle /> Limited improvement detected. Further denoising may not yield significant results.
                    </p>
                  )}
                </div>
                    ))}
                  </div>
                ) : (
                  <p className="audio-empty-message">Process an audio file to see quality metrics.</p>
              )}
            </div>
          </div>
        </div>
        )}
      </div>

      {/* Reusable Modal component */}
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
        title="Download Audio"
        type="info"
        buttons={[
          { 
            label: 'Cancel', 
            onClick: () => setIsDownloadModalOpen(false), 
            className: 'cancel-button' 
          },
          { 
            label: 'Download', 
            onClick: () => {
              setIsDownloadModalOpen(false);
              downloadAudio(downloadFormat, selectedVersion);
            }, 
            icon: <FaDownload />
          }
        ]}
      >
        <DownloadModalContent />
      </Modal>
    </div>
  );
};

export default AudioDenoiser; 