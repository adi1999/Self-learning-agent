#!/bin/bash
# Setup script for macOS workflow automation

set -e

echo "=========================================="
echo "Workflow Automation - macOS Setup"
echo "=========================================="
echo ""

# Check macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: This script is for macOS only"
    exit 1
fi

echo "1. Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo "Error: Python 3.8+ required (found $python_version)"
    exit 1
fi
echo "   ✓ Python $python_version"

echo ""
echo "2. Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   ✓ Created venv/"
else
    echo "   ✓ venv/ already exists"
fi

echo ""
echo "3. Activating virtual environment..."
source venv/bin/activate

echo ""
echo "4. Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "   ✓ Dependencies installed"

echo ""
echo "5. Installing Playwright browsers..."
playwright install chromium
echo "   ✓ Playwright ready"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: macOS Permissions Required"
echo ""
echo "You must grant the following permissions:"
echo ""
echo "1. Screen Recording:"
echo "   System Preferences → Security & Privacy → Screen Recording"
echo "   → Add Terminal (or your terminal app)"
echo ""
echo "2. Accessibility:"
echo "   System Preferences → Security & Privacy → Accessibility"
echo "   → Add Terminal (or your terminal app)"
echo ""
echo "Without these permissions, recording will fail!"
echo ""
echo "=========================================="
echo "Quick Start"
echo "=========================================="
echo ""
echo "# Activate environment"
echo "source venv/bin/activate"
echo ""
echo "# Record a workflow"
echo "python -m src.cli.record"
echo ""
echo "# Replay it (dumb mode)"
echo "python scripts/dumb_replay.py --session <session_id>"
echo ""
echo "=========================================="