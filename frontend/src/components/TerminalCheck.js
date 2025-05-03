import React, { useState, useEffect, useRef } from 'react';
import '../styles/TerminalCheck.css';

// Terminal messages to display in sequence
const TERMINAL_MESSAGES = [
  { text: 'DeepCleanse System Boot Sequence v1.0.2', delay: 500 },
  { text: 'Initializing hardware components...', delay: 800 },
  { text: 'Checking system resources...', delay: 1000 },
  { text: 'CPU: Online | RAM: Available | GPU: Detected', delay: 600 },
  { text: 'Loading TensorFlow environment...', delay: 1200 },
  { text: 'Initializing deep learning models...', delay: 1500 },
  { text: 'Checking for image denoising models...', delay: 1000 },
  { text: 'Image models loaded successfully.', delay: 800, success: true },
  { text: 'Checking for audio denoising models...', delay: 1000 },
  { text: 'Audio models loaded successfully.', delay: 800, success: true },
  { text: 'Checking network connectivity...', delay: 1200 },
  { text: 'Network status: ONLINE', delay: 800, success: true },
  { text: 'System check complete. All systems nominal.', delay: 1000, success: true },
  { text: 'Press Y to continue to DeepCleanse...', delay: 800, prompt: true }
];

const TerminalCheck = ({ onComplete, playSound }) => {
  const [messages, setMessages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [userInput, setUserInput] = useState('');
  const [showPrompt, setShowPrompt] = useState(false);
  const terminalRef = useRef(null);
  const inputRef = useRef(null);

  // Add new messages in sequence
  useEffect(() => {
    if (currentIndex < TERMINAL_MESSAGES.length) {
      const timer = setTimeout(() => {
        setMessages(prev => [...prev, TERMINAL_MESSAGES[currentIndex]]);
        playSound('terminal-type');
        
        if (TERMINAL_MESSAGES[currentIndex].prompt) {
          setShowPrompt(true);
          setTimeout(() => {
            if (inputRef.current) {
              inputRef.current.focus();
            }
          }, 100);
        } else {
          setCurrentIndex(i => i + 1);
        }
        
        // Scroll to bottom of terminal
        if (terminalRef.current) {
          terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
        }
      }, currentIndex === 0 ? 1000 : TERMINAL_MESSAGES[currentIndex - 1].delay);
      
      return () => clearTimeout(timer);
    }
  }, [currentIndex, playSound]);

  // Handle user input
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && showPrompt) {
      if (userInput.toLowerCase() === 'y' || userInput.toLowerCase() === 'yes') {
        playSound('terminal-type');
        setMessages(prev => [...prev, { text: `> ${userInput}`, user: true }]);
        setMessages(prev => [...prev, { text: 'Access granted. Launching DeepCleanse...', success: true }]);
        setUserInput('');
        setShowPrompt(false);
        
        // Delay before completing to show the final message
        setTimeout(() => {
          onComplete();
        }, 1500);
      } else {
        playSound('terminal-type');
        setMessages(prev => [...prev, { text: `> ${userInput}`, user: true }]);
        setMessages(prev => [...prev, { text: 'Invalid input. Press Y to continue...', error: true }]);
        setUserInput('');
      }
      
      // Scroll to bottom of terminal
      if (terminalRef.current) {
        terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
      }
    }
  };

  // Check network every 10 seconds
  useEffect(() => {
    let timer;
    if (showPrompt) {
      timer = setInterval(() => {
        const isOnline = navigator.onLine;
        if (!isOnline) {
          setMessages(prev => [...prev, { text: 'WARNING: Network connection lost. Retrying...', error: true }]);
        } else {
          setMessages(prev => [...prev, { text: 'Network status check: ONLINE', success: true }]);
        }
        
        // Scroll to bottom of terminal
        if (terminalRef.current) {
          terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
        }
      }, 10000);
    }
    
    return () => clearInterval(timer);
  }, [showPrompt]);

  return (
    <div className="terminal-container">
      <div className="terminal-header">
        <div className="terminal-controls">
          <span className="terminal-control terminal-close"></span>
          <span className="terminal-control terminal-minimize"></span>
          <span className="terminal-control terminal-maximize"></span>
        </div>
        <div className="terminal-title">DeepCleanse System Terminal</div>
      </div>
      
      <div className="terminal-body" ref={terminalRef}>
        {messages.map((msg, index) => (
          <div key={index} className={`terminal-line ${
            msg.success ? 'terminal-success' : msg.error ? 'terminal-error' : msg.user ? 'terminal-user-input' : ''
          }`}>
            {!msg.user && <span className="terminal-prompt">$</span>}
            {msg.text}
          </div>
        ))}
        
        {showPrompt && (
          <div className="terminal-input-line">
            <span className="terminal-prompt">$</span>
            <input
              type="text"
              ref={inputRef}
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              onKeyDown={handleKeyDown}
              className="terminal-input"
              autoFocus
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default TerminalCheck; 