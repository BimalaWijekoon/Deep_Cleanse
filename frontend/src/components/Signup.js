import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaEnvelope, FaLock, FaUser, FaUserPlus, FaBug, FaCheckCircle } from 'react-icons/fa';
import '../styles/Signup.css';
import Navbar from './Navbar';
import Modal from './Modal';
import axios from 'axios';

const Signup = () => {
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [error, setError] = useState('');
  const [showError, setShowError] = useState(false);
  const [showVerificationModal, setShowVerificationModal] = useState(false);
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [verificationCode, setVerificationCode] = useState('');
  const [isDebugMode, setIsDebugMode] = useState(false);
  const navigate = useNavigate();

  // For development - press "d" key to toggle debug mode
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.key === 'd' && e.ctrlKey) {
        setIsDebugMode(prev => !prev);
      }
    };
    
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, []);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      setShowError(true);
      return;
    }

    try {
      console.log('Submitting signup form:', formData);
      const response = await axios.post('http://localhost:5000/api/signup', {
        firstName: formData.firstName,
        lastName: formData.lastName,
        email: formData.email,
        password: formData.password
      });

      console.log('Signup response:', response.data);
      
      // Show verification modal after successful signup request
      setVerificationCode('');
      setShowVerificationModal(true);
      
    } catch (error) {
      console.error('Signup error:', error);
      setError(error.response?.data?.message || 'Signup failed. Please try again.');
      setShowError(true);
    }
  };

  const handleVerification = async () => {
    if (!verificationCode.trim()) {
      setError('Please enter the verification code');
      setShowError(true);
      return;
    }

    try {
      console.log('Submitting verification:', { email: formData.email, verificationCode });
      const response = await axios.post('http://localhost:5000/api/verify-email', {
        email: formData.email,
        verificationCode: verificationCode
      });

      console.log('Verification response:', response.data);
      
      // Hide verification modal
      setShowVerificationModal(false);
      
      // Show success modal
      setShowSuccessModal(true);
      
      // Redirect to login page after showing success message
      setTimeout(() => {
        setShowSuccessModal(false);
        navigate('/login');
      }, 2000);
      
    } catch (error) {
      console.error('Verification error:', error);
      setError(error.response?.data?.message || 'Verification failed. Please try again.');
      setShowError(true);
    }
  };

  // Debug functions
  const debugShowVerificationModal = () => {
    setVerificationCode('');
    setShowVerificationModal(true);
  };

  const debugShowSuccessModal = () => {
    setShowSuccessModal(true);
    setTimeout(() => {
      setShowSuccessModal(false);
    }, 2000);
  };

  return (
    <div className="signup-container">
      <Navbar />
      <div className="signup-content">
        <div className="signup-glass-card">
          <h2><FaUserPlus /> Sign Up</h2>
          {isDebugMode && (
            <div className="signup-debug-panel">
              <p className="signup-debug-info">
                Debug Mode: ON<br />
                Verification Modal: {showVerificationModal ? 'Visible' : 'Hidden'}<br />
                Success Modal: {showSuccessModal ? 'Visible' : 'Hidden'}<br />
              </p>
              <div className="signup-debug-buttons">
                <button type="button" onClick={debugShowVerificationModal} className="signup-debug-button">
                  <FaBug /> Show Verification
                </button>
                <button type="button" onClick={debugShowSuccessModal} className="signup-debug-button">
                  <FaBug /> Show Success
                </button>
              </div>
            </div>
          )}
          <form onSubmit={handleSubmit}>
            <div className="signup-form-group">
              <label htmlFor="firstName">First Name</label>
              <div className="signup-input-group">
                <FaUser className="signup-input-icon" />
                <input
                  type="text"
                  id="firstName"
                  name="firstName"
                  value={formData.firstName}
                  onChange={handleChange}
                  required
                  placeholder="Enter your first name"
                />
              </div>
            </div>
            <div className="signup-form-group">
              <label htmlFor="lastName">Last Name</label>
              <div className="signup-input-group">
                <FaUser className="signup-input-icon" />
                <input
                  type="text"
                  id="lastName"
                  name="lastName"
                  value={formData.lastName}
                  onChange={handleChange}
                  required
                  placeholder="Enter your last name"
                />
              </div>
            </div>
            <div className="signup-form-group">
              <label htmlFor="email">Email</label>
              <div className="signup-input-group">
                <FaEnvelope className="signup-input-icon" />
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                  placeholder="Enter your email"
                />
              </div>
            </div>
            <div className="signup-form-group">
              <label htmlFor="password">Password</label>
              <div className="signup-input-group">
                <FaLock className="signup-input-icon" />
                <input
                  type="password"
                  id="password"
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  required
                  placeholder="Enter your password"
                />
              </div>
            </div>
            <div className="signup-form-group">
              <label htmlFor="confirmPassword">Confirm Password</label>
              <div className="signup-input-group">
                <FaLock className="signup-input-icon" />
                <input
                  type="password"
                  id="confirmPassword"
                  name="confirmPassword"
                  value={formData.confirmPassword}
                  onChange={handleChange}
                  required
                  placeholder="Confirm your password"
                />
              </div>
            </div>
            <button type="submit" className="signup-submit-button">
              Sign Up
            </button>
          </form>
          <div className="signup-form-footer">
            <p>Already have an account? <a href="/login">Login</a></p>
          </div>
        </div>
      </div>

      {/* Error Modal */}
      <Modal
        isOpen={showError}
        onClose={() => setShowError(false)}
        title="Error"
        type="error"
        buttons={[
          {
            label: "OK",
            onClick: () => setShowError(false)
          }
        ]}
      >
        <p>{error}</p>
      </Modal>

      {/* Verification Modal */}
      <Modal
        isOpen={showVerificationModal}
        onClose={() => setShowVerificationModal(false)}
        title="Verify Your Email"
        type="info"
        buttons={[
          {
            label: "Verify",
            onClick: handleVerification,
            primary: true
          },
          {
            label: "Cancel",
            onClick: () => setShowVerificationModal(false)
          }
        ]}
      >
        <p>We've sent a verification code to your email address. Please enter it below:</p>
        <div className="signup-verification-input">
          <input 
            type="text" 
            value={verificationCode}
            onChange={(e) => setVerificationCode(e.target.value)}
            placeholder="Enter verification code"
            maxLength={6}
          />
        </div>
      </Modal>

      {/* Success Modal */}
      <Modal
        isOpen={showSuccessModal}
        onClose={() => setShowSuccessModal(false)}
        title="Success"
        type="success"
        hideCloseButton
      >
        <div className="signup-success-modal">
          <FaCheckCircle className="signup-success-icon" />
          <h3>Account Created Successfully!</h3>
          <p>Redirecting you to login...</p>
        </div>
      </Modal>
    </div>
  );
};

export default Signup; 