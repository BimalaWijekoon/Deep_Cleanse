import React, { useState, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { FaHome, FaSignInAlt, FaUserPlus, FaUser, FaSignOutAlt } from 'react-icons/fa';
import axios from 'axios';
import '../styles/Navbar.css';

const Navbar = ({ isLanding = false }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const [userData, setUserData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkUserLoggedIn = async () => {
      const userEmail = localStorage.getItem('userEmail');
      if (userEmail) {
        try {
          const response = await axios.get(`http://localhost:5000/api/user/${userEmail}`);
          setUserData(response.data.user);
        } catch (error) {
          console.error('Error fetching user data:', error);
          // Clear invalid user data
          localStorage.removeItem('userEmail');
          localStorage.removeItem('userName');
        }
      }
      setLoading(false);
    };

    checkUserLoggedIn();
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('userEmail');
    localStorage.removeItem('userName');
    setUserData(null);
    navigate('/login');
  };

  // Generate avatar initials from name
  const getInitials = () => {
    if (userData) {
      const firstInitial = userData.firstName ? userData.firstName.charAt(0) : '';
      const lastInitial = userData.lastName ? userData.lastName.charAt(0) : '';
      return (firstInitial + lastInitial).toUpperCase();
    }
    return 'U';
  };

  // Generate avatar color based on name
  const getAvatarColor = () => {
    if (!userData) return '#4dff88';
    
    const name = userData.firstName + userData.lastName;
    let hash = 0;
    for (let i = 0; i < name.length; i++) {
      hash = name.charCodeAt(i) + ((hash << 5) - hash);
    }
    
    let color = '#';
    for (let i = 0; i < 3; i++) {
      const value = (hash >> (i * 8)) & 0xFF;
      color += ('00' + value.toString(16)).substr(-2);
    }
    return color;
  };

  return (
    <nav className="navbar">
      <div className="navbar-content">
        <Link to="/" className="navbar-logo">
          <FaHome /> Encoders
        </Link>
        
        {isLanding && !loading && !userData && (
          <div className="navbar-limited-access">
            <span>Limited Access Mode</span>
          </div>
        )}
        
        <div className="navbar-links">
          {!loading && userData ? (
            <div className="navbar-user-profile">
              <div 
                className="navbar-user-avatar" 
                style={{ backgroundColor: getAvatarColor() }}
              >
                {getInitials()}
              </div>
              <span className="navbar-user-name">{userData.firstName} {userData.lastName}</span>
              <button onClick={handleLogout} className="navbar-logout-button">
                <FaSignOutAlt /> Logout
              </button>
            </div>
          ) : (
            <>
              <Link 
                to="/login" 
                className={`navbar-link ${location.pathname === '/login' ? 'navbar-active' : ''}`}
              >
                <FaSignInAlt /> Login
              </Link>
              <Link 
                to="/signup" 
                className={`navbar-link ${location.pathname === '/signup' ? 'navbar-active' : ''}`}
              >
                <FaUserPlus /> Sign Up
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar; 