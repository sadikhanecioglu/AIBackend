#!/bin/bash
# Setup script for the AI Gateway

echo "🚀 Setting up Multi-modal AI Gateway..."

# Create virtual environment (optional)
echo "📦 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment file
echo "⚙️ Setting up environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file from template"
    echo "⚠️  Please edit .env file with your API keys"
else
    echo "✅ .env file already exists"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p config
mkdir -p temp

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Edit .env file with your API keys"
echo "   2. Run: python main.py"
echo "   3. Visit: http://localhost:8000/docs"
echo ""
echo "🌐 Available endpoints:"
echo "   📊 Health: http://localhost:8000/health"
echo "   🔧 Providers: http://localhost:8000/providers" 
echo "   🤖 LLM: http://localhost:8000/llm"
echo "   🖼️ Image: http://localhost:8000/image"
echo "   📚 Docs: http://localhost:8000/docs"
echo "   🔗 WebSocket: ws://localhost:8000/voice"