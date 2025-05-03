import React from 'react';
import { FaTimes, FaCheckCircle, FaExclamationTriangle, FaInfoCircle } from 'react-icons/fa';
import '../styles/Modal.css';

const Modal = ({ 
  isOpen, 
  onClose, 
  title, 
  children, 
  type = 'default', 
  buttons = [],
  showCloseButton = true
}) => {
  if (!isOpen) return null;

  // Get appropriate icon based on modal type
  const getIcon = () => {
    switch (type) {
      case 'success':
        return <FaCheckCircle className="modal-icon success-icon" />;
      case 'warning':
        return <FaExclamationTriangle className="modal-icon warning-icon" />;
      case 'error':
        return <FaExclamationTriangle className="modal-icon error-icon" />;
      case 'info':
        return <FaInfoCircle className="modal-icon info-icon" />;
      default:
        return null;
    }
  };

  // Get appropriate class based on modal type
  const getModalClass = () => {
    return `modal-content glass-card ${type}-modal`;
  };

  return (
    <div className="modal-overlay" onClick={showCloseButton ? onClose : null}>
      <div className={getModalClass()} onClick={(e) => e.stopPropagation()}>
        {showCloseButton && (
          <button className="close-button-top-right" onClick={onClose}>
            <FaTimes />
          </button>
        )}
        
        {getIcon()}
        
        <div className="modal-header">
          <h3>{title}</h3>
        </div>
        
        <div className="modal-body">
          {children}
        </div>
        
        {buttons.length > 0 && (
          <div className="modal-actions">
            {buttons.map((button, index) => (
              <button 
                key={index} 
                className={`modal-button ${button.className || ''}`}
                onClick={button.onClick}
              >
                {button.icon && <span className="button-icon">{button.icon}</span>}
                {button.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Modal; 