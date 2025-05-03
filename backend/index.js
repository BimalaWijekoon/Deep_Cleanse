require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const cors = require('cors');
const helmet = require('helmet');
const nodemailer = require('nodemailer');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware setup
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));
app.use(helmet());

// MongoDB connection setup
const mongoURI = process.env.MONGODB_URI;
mongoose.connect(mongoURI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});
const db = mongoose.connection;
db.on('error', console.error.bind(console, 'MongoDB connection error:'));
db.once('open', () => console.log('Connected to MongoDB'));

// Nodemailer setup for sending emails
const transporter = nodemailer.createTransport({
  service: 'Gmail',
  auth: {
    user: process.env.EMAIL_USER,
    pass: process.env.EMAIL_PASS,
  },
});

// User Schema
const userSchema = new mongoose.Schema({
  firstName: { type: String, required: true },
  lastName: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  isVerified: { type: Boolean, default: true },
  createdAt: { type: Date, default: Date.now }
});

const User = mongoose.model('User', userSchema);

// Schema for denoised image sessions
const denoisedImageSchema = new mongoose.Schema({
  sessionName: { type: String, required: true },
  userEmail: { type: String, required: true },
  originalImage: { type: String, required: true },
  denoisedImages: [{
    round: { type: Number, required: true },
    image: { type: String, required: true },
    metrics: {
      noise_level: Number,
      var_reduction: Number,
      entropy_reduction: Number,
      noise_types: [String],
      psnr: Number,
      ssim: Number,
      low_improvement: Boolean,
      overall_noise_level: Number,
      noise_levels: { type: Map, of: Number }
    },
    timestamp: { type: Date, default: Date.now }
  }],
  analysisResults: {
    image_id: String,
    has_noise: Boolean,
    overall_noise_level: Number,
    noise_types: [String],
    noise_levels: { type: Map, of: Number },
    recommendations: String
  },
  createdAt: { type: Date, default: Date.now },
  lastUpdated: { type: Date, default: Date.now }
});

const DenoisedImage = mongoose.model('DenoisedImage', denoisedImageSchema);

// Schema for denoised audio sessions
const denoisedAudioSchema = new mongoose.Schema({
  sessionName: { type: String, required: true },
  userEmail: { type: String, required: true },
  originalAudio: { type: String, required: true },
  denoisedAudios: [{
    round: { type: Number, required: true },
    audio: { type: String, required: true },
    metrics: {
      snr: Number,
      noise_reduction: Number,
      spectrogram_improvement: Number,
      low_improvement: Boolean,
      clarity_score: Number,
      background_noise_reduction: Number
    },
    waveform: [Number],
    timestamp: { type: Date, default: Date.now }
  }],
  analysisResults: {
    audio_id: String,
    has_noise: Boolean,
    overall_noise_level: Number,
    noise_types: [String],
    noise_profile: String,
    recommendations: String
  },
  originalWaveform: [Number],
  createdAt: { type: Date, default: Date.now },
  lastUpdated: { type: Date, default: Date.now }
});

const DenoisedAudio = mongoose.model('DenoisedAudio', denoisedAudioSchema);

// In-memory pending users storage with expiry
const pendingUsers = new Map();

// Function to clean up expired pending users
const cleanupPendingUsers = () => {
  const now = Date.now();
  for (const [email, userData] of pendingUsers.entries()) {
    if (now - userData.timestamp > 10 * 60 * 1000) { // 10 minutes expiry
      pendingUsers.delete(email);
      console.log(`Expired verification for ${email} removed`);
    }
  }
};

// Run cleanup every 5 minutes
setInterval(cleanupPendingUsers, 5 * 60 * 1000);

// Routes
// Signup route
app.post('/api/signup', async (req, res) => {
  try {
    const { firstName, lastName, email, password } = req.body;
    
    // Check if user already exists in database
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ message: 'User already exists' });
    }

    // Check if user is pending verification
    if (pendingUsers.has(email)) {
      return res.status(400).json({ message: 'Verification already pending for this email' });
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 10);
    
    // Generate verification code
    const verificationCode = Math.floor(100000 + Math.random() * 900000).toString();
    
    // Store user data in memory
    pendingUsers.set(email, {
      firstName,
      lastName,
      email,
      password: hashedPassword,
      verificationCode,
      timestamp: Date.now()
    });

    // Send verification email
    const mailOptions = {
      from: process.env.EMAIL_USER,
      to: email,
      subject: 'Verify Your Email',
      html: `
        <h1>Email Verification</h1>
        <p>Your verification code is: <strong>${verificationCode}</strong></p>
        <p>This code will expire in 10 minutes.</p>
      `
    };

    await transporter.sendMail(mailOptions);
    
    res.status(201).json({ message: 'Verification code sent to your email. Please verify to complete registration.' });
  } catch (error) {
    console.error('Signup error:', error);
    res.status(500).json({ message: 'Error during signup process' });
  }
});

// Verify email route
app.post('/api/verify-email', async (req, res) => {
  try {
    const { email, verificationCode } = req.body;
    
    // Check if user is in pending verification
    if (!pendingUsers.has(email)) {
      return res.status(404).json({ message: 'No pending verification found for this email' });
    }

    const userData = pendingUsers.get(email);

    // Check verification code
    if (userData.verificationCode !== verificationCode) {
      return res.status(400).json({ message: 'Invalid verification code' });
    }

    // Create the user in database now that verification is complete
    const newUser = new User({
      firstName: userData.firstName,
      lastName: userData.lastName,
      email: userData.email,
      password: userData.password
    });
    
    await newUser.save();
    
    // Remove from pending users
    pendingUsers.delete(email);

    res.json({ message: 'Email verified and registration completed successfully' });
  } catch (error) {
    console.error('Verification error:', error);
    res.status(500).json({ message: 'Error during verification process' });
  }
});

// Login route
app.post('/api/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    
    // Check if user is in pending verification
    if (pendingUsers.has(email)) {
      return res.status(400).json({ message: 'Please complete email verification to login' });
    }
    
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    const validPassword = await bcrypt.compare(password, user.password);
    if (!validPassword) {
      return res.status(400).json({ message: 'Invalid password' });
    }

    res.json({ 
      message: 'Login successful',
      user: {
        firstName: user.firstName,
        lastName: user.lastName,
        email: user.email
      }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ message: 'Error logging in' });
  }
});

// Get user data by email
app.get('/api/user/:email', async (req, res) => {
  try {
    const { email } = req.params;
    
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    res.json({ 
      user: {
        firstName: user.firstName,
        lastName: user.lastName,
        email: user.email
      }
    });
  } catch (error) {
    console.error('Get user error:', error);
    res.status(500).json({ message: 'Error fetching user data' });
  }
});

// Route to request new verification code
app.post('/api/resend-verification', async (req, res) => {
  try {
    const { email } = req.body;
    
    // Check if user is in pending verification
    if (!pendingUsers.has(email)) {
      return res.status(404).json({ message: 'No pending verification found for this email' });
    }

    const userData = pendingUsers.get(email);
    
    // Generate new verification code
    const verificationCode = Math.floor(100000 + Math.random() * 900000).toString();
    
    // Update verification code
    userData.verificationCode = verificationCode;
    userData.timestamp = Date.now(); // Reset expiry timer
    pendingUsers.set(email, userData);

    // Send verification email
    const mailOptions = {
      from: process.env.EMAIL_USER,
      to: email,
      subject: 'Verify Your Email',
      html: `
        <h1>Email Verification</h1>
        <p>Your new verification code is: <strong>${verificationCode}</strong></p>
        <p>This code will expire in 10 minutes.</p>
      `
    };

    await transporter.sendMail(mailOptions);
    
    res.json({ message: 'New verification code sent to your email' });
  } catch (error) {
    console.error('Resend verification error:', error);
    res.status(500).json({ message: 'Error sending new verification code' });
  }
});

// Routes for saving denoised image sessions
app.post('/api/image/save-session', async (req, res) => {
  try {
    const { sessionName, userEmail, originalImage, denoisedImages, analysisResults } = req.body;
    
    // Check if session already exists for this user
    let session = await DenoisedImage.findOne({ sessionName, userEmail });
    
    if (session) {
      // Update existing session
      session.denoisedImages = denoisedImages;
      // Only update analysis results if provided
      if (analysisResults) {
        session.analysisResults = analysisResults;
      }
      session.lastUpdated = Date.now();
      await session.save();
      res.status(200).json({ message: 'Session updated successfully', sessionId: session._id });
    } else {
      // Create new session
      const newSession = new DenoisedImage({
        sessionName,
        userEmail,
        originalImage,
        denoisedImages,
        analysisResults: analysisResults || null
      });
      
      await newSession.save();
      res.status(201).json({ message: 'Session saved successfully', sessionId: newSession._id });
    }
  } catch (error) {
    console.error('Error saving image session:', error);
    res.status(500).json({ message: 'Error saving session' });
  }
});

// Get user's image sessions
app.get('/api/image/sessions/:email', async (req, res) => {
  try {
    const { email } = req.params;
    
    const sessions = await DenoisedImage.find({ userEmail: email }).sort({ lastUpdated: -1 });
    
    res.status(200).json({ sessions });
  } catch (error) {
    console.error('Error fetching image sessions:', error);
    res.status(500).json({ message: 'Error fetching sessions' });
  }
});

// Advanced search for image sessions by metrics criteria
app.post('/api/image/sessions/search', async (req, res) => {
  try {
    const { email, criteria } = req.body;
    
    // Build query based on provided criteria
    const query = { userEmail: email };
    
    // Add criteria filters if provided
    if (criteria) {
      if (criteria.noise_types && criteria.noise_types.length > 0) {
        query['analysisResults.noise_types'] = { $in: criteria.noise_types };
      }
      
      if (criteria.min_noise_level !== undefined) {
        query['analysisResults.overall_noise_level'] = { $gte: criteria.min_noise_level };
      }
      
      if (criteria.max_noise_level !== undefined) {
        if (query['analysisResults.overall_noise_level']) {
          query['analysisResults.overall_noise_level'].$lte = criteria.max_noise_level;
        } else {
          query['analysisResults.overall_noise_level'] = { $lte: criteria.max_noise_level };
        }
      }
      
      if (criteria.min_improvement !== undefined) {
        query['denoisedImages.metrics.var_reduction'] = { $gte: criteria.min_improvement };
      }
      
      if (criteria.min_psnr !== undefined) {
        query['denoisedImages.metrics.psnr'] = { $gte: criteria.min_psnr };
      }
      
      if (criteria.min_ssim !== undefined) {
        query['denoisedImages.metrics.ssim'] = { $gte: criteria.min_ssim };
      }
    }
    
    const sessions = await DenoisedImage.find(query).sort({ lastUpdated: -1 });
    
    res.status(200).json({ sessions });
  } catch (error) {
    console.error('Error searching image sessions:', error);
    res.status(500).json({ message: 'Error searching sessions' });
  }
});

// Get specific image session
app.get('/api/image/session/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const session = await DenoisedImage.findById(id);
    
    if (!session) {
      return res.status(404).json({ message: 'Session not found' });
    }
    
    res.status(200).json({ session });
  } catch (error) {
    console.error('Error fetching image session:', error);
    res.status(500).json({ message: 'Error fetching session' });
  }
});

// Routes for saving denoised audio sessions
app.post('/api/audio/save-session', async (req, res) => {
  try {
    const { sessionName, userEmail, originalAudio, denoisedAudios, analysisResults, originalWaveform } = req.body;
    
    // Check if session already exists for this user
    let session = await DenoisedAudio.findOne({ sessionName, userEmail });
    
    if (session) {
      // Update existing session
      session.denoisedAudios = denoisedAudios;
      // Only update analysis results if provided
      if (analysisResults) {
        session.analysisResults = analysisResults;
      }
      // Only update original waveform if provided
      if (originalWaveform && originalWaveform.length > 0) {
        session.originalWaveform = originalWaveform;
      }
      session.lastUpdated = Date.now();
      await session.save();
      res.status(200).json({ message: 'Session updated successfully', sessionId: session._id });
    } else {
      // Create new session
      const newSession = new DenoisedAudio({
        sessionName,
        userEmail,
        originalAudio,
        denoisedAudios,
        analysisResults: analysisResults || null,
        originalWaveform: originalWaveform || []
      });
      
      await newSession.save();
      res.status(201).json({ message: 'Session saved successfully', sessionId: newSession._id });
    }
  } catch (error) {
    console.error('Error saving audio session:', error);
    res.status(500).json({ message: 'Error saving session' });
  }
});

// Get user's audio sessions
app.get('/api/audio/sessions/:email', async (req, res) => {
  try {
    const { email } = req.params;
    
    const sessions = await DenoisedAudio.find({ userEmail: email }).sort({ lastUpdated: -1 });
    
    res.status(200).json({ sessions });
  } catch (error) {
    console.error('Error fetching audio sessions:', error);
    res.status(500).json({ message: 'Error fetching sessions' });
  }
});

// Advanced search for audio sessions by metrics criteria
app.post('/api/audio/sessions/search', async (req, res) => {
  try {
    const { email, criteria } = req.body;
    
    // Build query based on provided criteria
    const query = { userEmail: email };
    
    // Add criteria filters if provided
    if (criteria) {
      if (criteria.noise_types && criteria.noise_types.length > 0) {
        query['analysisResults.noise_types'] = { $in: criteria.noise_types };
      }
      
      if (criteria.min_noise_level !== undefined) {
        query['analysisResults.overall_noise_level'] = { $gte: criteria.min_noise_level };
      }
      
      if (criteria.max_noise_level !== undefined) {
        if (query['analysisResults.overall_noise_level']) {
          query['analysisResults.overall_noise_level'].$lte = criteria.max_noise_level;
        } else {
          query['analysisResults.overall_noise_level'] = { $lte: criteria.max_noise_level };
        }
      }
      
      if (criteria.min_noise_reduction !== undefined) {
        query['denoisedAudios.metrics.noise_reduction'] = { $gte: criteria.min_noise_reduction };
      }
      
      if (criteria.min_snr !== undefined) {
        query['denoisedAudios.metrics.snr'] = { $gte: criteria.min_snr };
      }
      
      if (criteria.min_clarity !== undefined) {
        query['denoisedAudios.metrics.clarity_score'] = { $gte: criteria.min_clarity };
      }
    }
    
    const sessions = await DenoisedAudio.find(query).sort({ lastUpdated: -1 });
    
    res.status(200).json({ sessions });
  } catch (error) {
    console.error('Error searching audio sessions:', error);
    res.status(500).json({ message: 'Error searching sessions' });
  }
});

// Get specific audio session
app.get('/api/audio/session/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const session = await DenoisedAudio.findById(id);
    
    if (!session) {
      return res.status(404).json({ message: 'Session not found' });
    }
    
    res.status(200).json({ session });
  } catch (error) {
    console.error('Error fetching audio session:', error);
    res.status(500).json({ message: 'Error fetching session' });
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
}); 