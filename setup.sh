#!/bin/bash

# Setup script for AI Video Analyzer on Unix-like systems (macOS/Linux)

echo "🎥 AI Video Analyzer Setup Script"
echo "================================="

# Check if Python 3.9+ is installed
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Found Python $python_version"

# Check if Node.js is installed
echo "Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18 or higher."
    exit 1
fi

node_version=$(node --version)
echo "✅ Found Node.js $node_version"

# Check if Rust is installed (for Tauri)
echo "Checking Rust installation..."
if ! command -v rustc &> /dev/null; then
    echo "⚠️  Rust is not installed. Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

rust_version=$(rustc --version)
echo "✅ Found Rust: $rust_version"

# Set up Python backend
echo ""
echo "Setting up Python backend..."
cd backend

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Backend setup complete!"

# Set up frontend
echo ""
echo "Setting up React frontend..."
cd ../frontend

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm install

# Install Tauri CLI if not present
echo "Installing Tauri CLI..."
npm install -g @tauri-apps/cli

echo "✅ Frontend setup complete!"

# Create startup scripts
echo ""
echo "Creating startup scripts..."
cd ..

# Backend startup script
cat > start_backend.sh << 'EOF'
#!/bin/bash
cd backend
source venv/bin/activate
echo "Starting AI Video Analyzer Backend..."
python main.py
EOF

# Frontend startup script
cat > start_frontend.sh << 'EOF'
#!/bin/bash
cd frontend
echo "Starting AI Video Analyzer Frontend..."
npm run tauri dev
EOF

# Make scripts executable
chmod +x start_backend.sh
chmod +x start_frontend.sh

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the application:"
echo "1. Open a terminal and run: ./start_backend.sh"
echo "2. Open another terminal and run: ./start_frontend.sh"
echo ""
echo "The application will be available on your desktop!"