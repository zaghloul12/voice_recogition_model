#!/bin/bash
# run.sh - Quick start script with folder support

echo "🤖 Robot Assistant Quick Start"
echo "=============================="

# Default faces directory
FACES_DIR=${1:-"known_faces"}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed"
    exit 1
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."
pip3 show opencv-python > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "📦 Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Check faces directory
if [ ! -d "$FACES_DIR" ]; then
    echo "⚠️  Faces directory '$FACES_DIR' not found!"
    echo "📁 Please create it with subfolders for each person."
    echo "   Example: $FACES_DIR/John/photo1.jpg"
    exit 1
fi

# List people
echo "📋 People in database:"
python3 list_people.py --faces-dir "$FACES_DIR"

# Run the robot
echo ""
echo "🚀 Starting robot assistant..."
python3 main.py --faces-dir "$FACES_DIR" "$@"