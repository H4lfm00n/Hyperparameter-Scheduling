#!/bin/bash

# Automatic Hyperparameter Scheduling Library Installation Script

echo "🚀 Installing Automatic Hyperparameter Scheduling Library"
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python version $python_version is too old. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment (optional)
read -p "🤔 Do you want to create a virtual environment? (y/n): " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created and activated"
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install the library in development mode
echo "🔧 Installing library in development mode..."
pip install -e .

# Run tests
echo "🧪 Running tests..."
python -m pytest tests/ -v

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📚 Next steps:"
echo "   1. Check out the examples in the 'examples/' directory"
echo "   2. Run 'python examples/basic_usage.py' for a quick demo"
echo "   3. Run 'python examples/transfer_learning_demo.py' for advanced features"
echo ""
echo "📖 Documentation: See README.md for detailed usage instructions"
echo ""
echo "🔧 To activate virtual environment (if created):"
echo "   source venv/bin/activate"
