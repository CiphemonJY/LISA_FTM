#!/bin/bash
# LISA Federated Learning - One-line installer
# Usage: curl -sL https://lisa.ciphemon.ai/install | bash -s JOIN_CODE

JOIN_CODE="${1:-}"
if [ -z "$JOIN_CODE" ]; then
    echo "Usage: curl -sL https://lisa.ciphemon.ai/install | bash -s YOUR_CODE"
    echo "   Or: curl -sL https://lisa.ciphemon.ai/install | bash -s"
    echo ""
    echo "Get a join code from your federated network operator"
    exit 1
fi

echo "🤝 Installing LISA Federated Client..."
echo "   Join Code: $JOIN_CODE"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 required"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install --quiet torch transformers peft requests

# Download client
CLIENT_URL="https://lisa.ciphemon.ai/client/lisa_client_easy.py"
curl -sL "$CLIENT_URL" -o /tmp/lisa_client.py

if [ $? -eq 0 ]; then
    echo "✅ Client downloaded"
    echo ""
    echo "🚀 Starting training..."
    python3 /tmp/lisa_client.py --join "$JOIN_CODE"
else
    echo "⚠️  Download failed, trying direct connect..."
    python3 /tmp/lisa_client_easy.py --join "$JOIN_CODE" 2>/dev/null || \
    echo "Please install manually: pip3 install torch transformers peft requests"
fi
