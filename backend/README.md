# FacenRoll Backend

This is the backend server for the FacenRoll application. It handles user authentication, email verification, and data management.

## Setup Instructions

1. Install dependencies:
```bash
npm install
```

2. Create a `.env` file in the root directory with the following variables:
```
MONGODB_URI=mongodb://localhost:27017/FacenRoll
PORT=5000
EMAIL_USER=your-email@gmail.com
EMAIL_PASS=your-email-app-password
```

3. Start the server:
```bash
npm start
```

For development with auto-reload:
```bash
npm run dev
```

## API Endpoints

- POST `/api/signup` - Register a new user
- POST `/api/verify-email` - Verify user's email with verification code
- POST `/api/login` - Login user

## Dependencies

- Express.js
- MongoDB with Mongoose
- Nodemailer for email verification
- bcrypt for password hashing
- cors for cross-origin requests
- helmet for security headers 