import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaEnvelope, FaLock, FaSignInAlt, FaExclamationTriangle } from 'react-icons/fa';
import '../styles/Login.css';
import Navbar from './Navbar';
import Modal from './Modal';
import axios from 'axios';

const Login = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [error, setError] = useState('');
  const [showError, setShowError] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const [showSkipWarning, setShowSkipWarning] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      console.log('Submitting login form:', { email: formData.email });
      const response = await axios.post('http://localhost:5000/api/login', formData);
      
      console.log('Login response:', response.data);
      
      // Store user information
      localStorage.setItem('userEmail', formData.email);
      if (response.data.user) {
        localStorage.setItem('userName', `${response.data.user.firstName} ${response.data.user.lastName}`);
      }
      
      // Show success message
      setShowSuccess(true);
      
      // Navigate after delay
      setTimeout(() => {
        setShowSuccess(false);
        navigate('/landing');
      }, 1500);
    } catch (error) {
      console.error('Login error:', error);
      setError(error.response?.data?.message || 'Login failed. Please try again.');
      setShowError(true);
    }
  };

  const handleSkipLogin = () => {
    setShowSkipWarning(true);
  };

  const proceedToLanding = () => {
    setShowSkipWarning(false);
    navigate('/landing');
  };

  return (
    <div className="login-container">
      <Navbar />
      <div className="login-content">
        <div className="login-glass-card">
          <h2><FaSignInAlt /> Login</h2>
          <form onSubmit={handleSubmit}>
            <div className="login-form-group">
              <label htmlFor="email">Email</label>
              <div className="login-input-group">
                <FaEnvelope className="login-input-icon" />
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
            <div className="login-form-group">
              <label htmlFor="password">Password</label>
              <div className="login-input-group">
                <FaLock className="login-input-icon" />
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
            <button type="submit" className="login-submit-button">
              Login
            </button>
          </form>
          <div className="login-form-footer">
            <p>Don't have an account? <a href="/signup">Sign up</a></p>
            <p className="login-skip-text">
              <a href="#" onClick={(e) => { e.preventDefault(); handleSkipLogin(); }}>Skip login and continue to app</a>
            </p>
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

      {/* Success Modal */}
      <Modal
        isOpen={showSuccess}
        onClose={() => {}}
        title="Login Successful!"
        type="success"
        hideCloseButton
      >
        <div className="login-success-modal">
          <p>Welcome back! Redirecting to your dashboard...</p>
        </div>
      </Modal>

      {/* Skip Login Warning Modal */}
      <Modal
        isOpen={showSkipWarning}
        onClose={() => setShowSkipWarning(false)}
        title="Limited Access"
        type="warning"
        buttons={[
          {
            label: "Cancel",
            onClick: () => setShowSkipWarning(false)
          },
          {
            label: "Skip login and continue",
            onClick: proceedToLanding,
            primary: true
          }
        ]}
      >
        <div className="login-warning-modal">
          <p>
            If you skip the login, you cannot download or save your works. Please <a href="/signup" className="login-signup-link">signup</a> for that.
          </p>
        </div>
      </Modal>
    </div>
  );
};

export default Login; 