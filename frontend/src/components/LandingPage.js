import React, { useState, useEffect } from 'react';
import '../styles/LandingPage.css';
import { FaImage, FaMusic, FaRegLightbulb, FaHistory, FaAngleRight, FaInfoCircle } from 'react-icons/fa';
import Navbar from './Navbar';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const LandingPage = ({ onSelectImage, onSelectAudio }) => {
  const navigate = useNavigate();
  const [userEmail, setUserEmail] = useState(null);
  const [imageHistory, setImageHistory] = useState([]);
  const [audioHistory, setAudioHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [activeTab, setActiveTab] = useState('image');
  const [isExpanded, setIsExpanded] = useState(false);

  // Check if user is logged in and fetch session history
  useEffect(() => {
    const email = localStorage.getItem('userEmail');
    if (email) {
      setUserEmail(email);
      fetchUserHistory(email);
      setShowHistory(true);
    }
  }, []);

  // Fetch user's denoising history from the backend
  const fetchUserHistory = async (email) => {
    setLoading(true);
    try {
      // Fetch image sessions
      const imageResponse = await axios.get(`http://localhost:5000/api/image/sessions/${email}`);
      setImageHistory(imageResponse.data.sessions || []);
      
      // Fetch audio sessions
      const audioResponse = await axios.get(`http://localhost:5000/api/audio/sessions/${email}`);
      setAudioHistory(audioResponse.data.sessions || []);
    } catch (error) {
      console.error('Error fetching user history:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleImageSelect = (e) => {
    e.preventDefault(); // Prevent default button behavior
    onSelectImage();
  };

  const handleAudioSelect = (e) => {
    e.preventDefault(); // Prevent default button behavior
    onSelectAudio();
  };

  // Open a specific image denoising session
  const openImageSession = (session) => {
    localStorage.setItem('currentImageSession', JSON.stringify(session));
    navigate('/image');
  };

  // Open a specific audio denoising session
  const openAudioSession = (session) => {
    localStorage.setItem('currentAudioSession', JSON.stringify(session));
    navigate('/audio');
  };

  // Format date for display
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Extract the original filename from session name
  const getSessionDisplayName = (sessionName) => {
    // Remove timestamp and user part from session name
    const parts = sessionName.split('_');
    if (parts.length >= 3) {
      // The first part(s) should be the original filename
      return parts.slice(0, parts.length - 2).join('_');
    }
    return sessionName;
  };

  return (
    <div className="landing-container">
      <Navbar isLanding={true} />
      <div className="landing-content-wrapper">
        {/* History Panel - Hover to expand */}
        {userEmail && (
          <div 
            className={`landing-history-panel ${isExpanded ? 'expanded' : 'collapsed'}`}
            onMouseEnter={() => setIsExpanded(true)}
            onMouseLeave={() => setIsExpanded(false)}
          >
            <div className="history-panel-content">
              <div className="history-header">
                <div className="history-icon-container pulse-animation">
                  <FaHistory className="history-icon" />
                </div>
                <h3>Your Recent Work</h3>
              </div>
              
              {isExpanded && (
                <>
                  <div className="history-tabs">
                    <button 
                      className={`history-tab ${activeTab === 'image' ? 'active' : ''}`}
                      onClick={() => setActiveTab('image')}
                    >
                      <FaImage /> Image Sessions
                    </button>
                    <button 
                      className={`history-tab ${activeTab === 'audio' ? 'active' : ''}`}
                      onClick={() => setActiveTab('audio')}
                    >
                      <FaMusic /> Audio Sessions
                    </button>
                  </div>
                  
                  <div className="history-content">
                    {loading ? (
                      <div className="history-loading">
                        <div className="history-spinner"></div>
                        <p>Loading your sessions...</p>
                      </div>
                    ) : activeTab === 'image' ? (
                      <div className="history-list fade-in-animation-delayed">
                        {imageHistory.length > 0 ? (
                          imageHistory.map((session, index) => (
                            <div 
                              key={session._id} 
                              className="history-item scale-in-animation" 
                              onClick={() => openImageSession(session)}
                              style={{ animationDelay: `${0.1 * index}s` }}
                            >
                              <div className="history-item-preview">
                                {session.originalImage ? (
                                  <img 
                                    src={session.originalImage} 
                                    alt="Preview" 
                                    className="history-item-thumbnail"
                                  />
                                ) : (
                                  <FaImage className="history-preview-icon" />
                                )}
                              </div>
                              <div className="history-item-details">
                                <h4>{getSessionDisplayName(session.sessionName)}</h4>
                                <p>Updated: {formatDate(session.lastUpdated)}</p>
                                <p>Rounds: {session.denoisedImages.length}</p>
                              </div>
                              <FaAngleRight className="history-item-arrow" />
                            </div>
                          ))
                        ) : (
                          <div className="history-empty">
                            <FaInfoCircle />
                            <p>No image denoising sessions found. Try processing an image!</p>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="history-list fade-in-animation-delayed">
                        {audioHistory.length > 0 ? (
                          audioHistory.map((session, index) => (
                            <div 
                              key={session._id} 
                              className="history-item scale-in-animation" 
                              onClick={() => openAudioSession(session)}
                              style={{ animationDelay: `${0.1 * index}s` }}
                            >
                              <div className="history-item-preview">
                                <FaMusic className="history-preview-icon" />
                              </div>
                              <div className="history-item-details">
                                <h4>{getSessionDisplayName(session.sessionName)}</h4>
                                <p>Updated: {formatDate(session.lastUpdated)}</p>
                                <p>Rounds: {session.denoisedAudios.length}</p>
                              </div>
                              <FaAngleRight className="history-item-arrow" />
                            </div>
                          ))
                        ) : (
                          <div className="history-empty">
                            <FaInfoCircle />
                            <p>No audio denoising sessions found. Try processing an audio file!</p>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </>
              )}
              
              {!isExpanded && (
                <div className="collapsed-indicator">
                  <div className="vertical-text">Recent Work</div>
                  <div className="session-count">
                    <span>{imageHistory.length + audioHistory.length}</span>
                    <small>sessions</small>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Main Content Panel - Adjusted width based on sidebar state */}
        <div className={`landing-main-panel ${!userEmail ? 'full-width' : isExpanded ? 'with-sidebar' : 'with-mini-sidebar'}`}>
          <div className="glass-panel">
            <div className="logo-container">
              <FaRegLightbulb className="logo-icon pulse-animation" />
              <h1 className="slide-in-animation">DeepCleanse</h1>
            </div>

            <h2 className="fade-in-animation">Advanced AI-Powered Denoising</h2>
            
            <p className="description fade-in-animation-delayed">
              Remove noise and enhance your media with state-of-the-art deep learning technology.
              Choose the type of media you want to process.
            </p>
            
            <div className="options-container scale-in-animation">
              <div className="option-card" onClick={handleImageSelect}>
                <div className="option-icon-container">
                  <FaImage className="option-icon" />
                </div>
                <h3>Image Denoising</h3>
                <p>Remove noise from photos and enhance image quality</p>
                <div className="option-details">
                  <ul>
                    <li>Remove salt & pepper noise</li>
                    <li>Reduce gaussian noise</li>
                    <li>Enhance image clarity</li>
                    <li>Analyze noise patterns</li>
                  </ul>
                </div>
                <button className="option-button" onClick={handleImageSelect}>Open Image Denoiser</button>
              </div>
              
              <div className="option-card" onClick={handleAudioSelect}>
                <div className="option-icon-container">
                  <FaMusic className="option-icon" />
                </div>
                <h3>Audio Denoising</h3>
                <p>Clean up audio recordings and enhance sound quality</p>
                <div className="option-details">
                  <ul>
                    <li>Remove background noise</li>
                    <li>Eliminate hum and buzz</li>
                    <li>Enhance speech clarity</li>
                    <li>Improve audio quality</li>
                  </ul>
                </div>
                <button className="option-button" onClick={handleAudioSelect}>Open Audio Denoiser</button>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Main Footer */}
      <footer className="main-footer">
        <p>All Rights Reserved | <span className="footer-brand">DeepCleanse</span> &copy; {new Date().getFullYear()} | Powered by Deep Learning</p>
      </footer>
      
      {/* Background animation elements */}
      <div className="background-animation">
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
      </div>
    </div>
  );
};

export default LandingPage; 