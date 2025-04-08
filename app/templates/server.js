const express = require('express');
const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const cors = require('cors');
const dotenv = require('dotenv');

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public')); // Serve static files from 'public' directory

// MongoDB Connection
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/career_compass')
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.error('MongoDB connection error:', err));

// User Schema
const userSchema = new mongoose.Schema({
  email: { 
    type: String, 
    required: true, 
    unique: true,
    trim: true,
    lowercase: true
  },
  first_name: { 
    type: String, 
    required: true,
    trim: true
  },
  last_name: { 
    type: String, 
    required: true,
    trim: true
  },
  password: { 
    type: String, 
    required: true 
  },
  created_at: { 
    type: Date, 
    default: Date.now 
  },
  profile: {
    current_job_title: String,
    industry: String,
    skills: [String],
    desired_job_titles: [String],
    desired_locations: [String],
    min_salary: Number,
    remote_preference: Boolean,
    flexible_hours_preference: Boolean
  }
});

// User Model
const User = mongoose.model('User', userSchema);

// Register endpoint
app.post('/api/users/register', async (req, res) => {
  try {
    // Check if user already exists
    const userExists = await User.findOne({ email: req.body.email });
    if (userExists) {
      return res.status(400).json({ 
        success: false, 
        message: 'Email is already registered' 
      });
    }

    // Hash the password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(req.body.password, salt);

    // Create new user
    const newUser = new User({
      email: req.body.email,
      first_name: req.body.first_name,
      last_name: req.body.last_name,
      password: hashedPassword
    });

    // Save to database
    const savedUser = await newUser.save();
    
    // Return user data (without password)
    res.status(201).json({
      success: true,
      user: {
        id: savedUser._id,
        email: savedUser.email,
        first_name: savedUser.first_name,
        last_name: savedUser.last_name
      }
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Server error during registration' 
    });
  }
});

// Login endpoint
app.post('/api/users/login', async (req, res) => {
  try {
    // Find user by email
    const user = await User.findOne({ email: req.body.email });
    if (!user) {
      return res.status(400).json({ 
        success: false, 
        message: 'Invalid email or password' 
      });
    }

    // Verify password
    const validPassword = await bcrypt.compare(req.body.password, user.password);
    if (!validPassword) {
      return res.status(400).json({ 
        success: false, 
        message: 'Invalid email or password' 
      });
    }

    // Return user data (without password)
    res.status(200).json({
      success: true,
      user: {
        id: user._id,
        email: user.email,
        first_name: user.first_name,
        last_name: user.last_name,
        profile: user.profile
      }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Server error during login' 
    });
  }
});

// Update profile endpoint
app.put('/api/users/:id/profile', async (req, res) => {
  try {
    const updatedUser = await User.findByIdAndUpdate(
      req.params.id,
      { profile: req.body },
      { new: true, runValidators: true }
    ).select('-password');

    if (!updatedUser) {
      return res.status(404).json({ 
        success: false, 
        message: 'User not found' 
      });
    }

    res.status(200).json({
      success: true,
      user: updatedUser
    });
  } catch (error) {
    console.error('Profile update error:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Server error during profile update' 
    });
  }
});

// Get user profile endpoint
app.get('/api/users/:id', async (req, res) => {
  try {
    const user = await User.findById(req.params.id).select('-password');
    
    if (!user) {
      return res.status(404).json({ 
        success: false, 
        message: 'User not found' 
      });
    }

    res.status(200).json({
      success: true,
      user
    });
  } catch (error) {
    console.error('Get user error:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Server error retrieving user data' 
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
