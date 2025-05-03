import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import './styles/App.css';
import LandingPage from './components/LandingPage';
import ImageDenoiser from './components/ImageDenoiser';
import AudioDenoiser from './components/AudioDenoiser';
import TerminalCheck from './components/TerminalCheck';
import Login from './components/Login';
import Signup from './components/Signup';

function App() {
  // State to track the current mode
  const [soundsLoaded, setSoundsLoaded] = useState(false);
  const [showSkipModal, setShowSkipModal] = useState(false);
  
  // Preload sounds to ensure they're always available
  useEffect(() => {
    const soundFiles = [
      '/sounds/terminal-type.mp3'
    ];
    
    const preloadedSounds = {};
    let loadedCount = 0;
    
    soundFiles.forEach(soundFile => {
      const audio = new Audio();
      audio.src = soundFile;
      
      // Listen for when the audio is ready to play
      audio.addEventListener('canplaythrough', () => {
        loadedCount++;
        if (loadedCount === soundFiles.length) {
          setSoundsLoaded(true);
        }
      }, { once: true });
      
      // Store the audio object in our preloaded sounds map
      preloadedSounds[soundFile] = audio;
    });
    
    // Attach to window for global access
    window.preloadedSounds = preloadedSounds;
    
    // If sounds don't load within 3 seconds, continue anyway
    const timeout = setTimeout(() => {
      if (!soundsLoaded) {
        setSoundsLoaded(true);
      }
    }, 3000);
    
    return () => clearTimeout(timeout);
  }, []);
  
  // Play sound utility function - only for terminal-type.mp3
  const playSound = (soundType) => {
    // Only process terminal-type sound, silently ignore other sound requests
    if (soundType !== 'terminal-type') return;
    
    try {
      const soundFile = '/sounds/terminal-type.mp3';
      const volume = 0.2;
      
      // Try to use preloaded sound if available
      if (window.preloadedSounds && window.preloadedSounds[soundFile]) {
        const sound = window.preloadedSounds[soundFile];
        sound.currentTime = 0; // Reset to beginning
        sound.volume = volume;
        sound.play().catch(error => {
          // Silently handle any play errors
          console.debug('Audio playback not allowed:', error);
        });
      } else {
        // Fallback to creating a new Audio instance
        const sound = new Audio(soundFile);
        sound.volume = volume;
        sound.play().catch(error => {
          // Silently handle any play errors
          console.debug('Audio playback not allowed:', error);
        });
      }
    } catch (err) {
      // Silently handle errors
      console.debug('Sound play error:', err);
    }
  };

  const handleSkipLogin = () => {
    setShowSkipModal(true);
  };

  // Wrapper components to provide navigation functionality
  const LandingPageWithNavigation = () => {
    const navigate = useNavigate();
    return (
      <LandingPage 
        onSelectImage={() => navigate('/image')} 
        onSelectAudio={() => navigate('/audio')}
      />
    );
  };

  const ImageDenoiserWithNavigation = () => {
    const navigate = useNavigate();
    const isLoggedIn = localStorage.getItem('userEmail') !== null;
    return <ImageDenoiser goBack={() => navigate('/landing')} playSound={() => {}} isLoggedIn={isLoggedIn} />;
  };

  const AudioDenoiserWithNavigation = () => {
    const navigate = useNavigate();
    const isLoggedIn = localStorage.getItem('userEmail') !== null;
    return <AudioDenoiser goBack={() => navigate('/landing')} playSound={() => {}} isLoggedIn={isLoggedIn} />;
  };

  const TerminalCheckWithNavigation = () => {
    const navigate = useNavigate();
    return <TerminalCheck onComplete={() => navigate('/login')} playSound={playSound} />;
  };

  return (
    <Router>
      <div className="app-container">
        <Routes>
          <Route path="/" element={<Navigate to="/terminal" />} />
          <Route path="/terminal" element={<TerminalCheckWithNavigation />} />
          <Route path="/login" element={<Login playSound={() => {}} />} />
          <Route path="/signup" element={<Signup playSound={() => {}} />} />
          <Route path="/landing" element={<LandingPageWithNavigation />} />
          <Route path="/image" element={<ImageDenoiserWithNavigation />} />
          <Route path="/audio" element={<AudioDenoiserWithNavigation />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App; 