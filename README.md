# Snap Labs AI Chatbot

An intelligent chatbot for Snap Labs website that helps users learn about our services and book consultation calls.

## Features

- 🤖 Agentic framework using LangGraph for complex conversations
- 📚 RAG (Retrieval Augmented Generation) for accurate company information
- 📅 Integration with Outlook calendar for booking calls
- 💬 Modern chat interface using Chainlit
- 🔄 Seamless embedding in the Snap Labs website

## Prerequisites

1. Install Ollama:
```bash
curl https://ollama.ai/install.sh | sh
```

2. Pull the Mistral model:
```bash
ollama pull mistral
```

## Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create knowledge base directory:
```bash
mkdir -p data/knowledge_base
```

4. Add your company documents to `data/knowledge_base/`

## Running the Chatbot

1. Start the Chainlit server:
```bash
chainlit run app.py
```

2. The chatbot will be available at `http://localhost:8000`

## Integration with Next.js Website

1. Install the Chainlit React component:
```bash
cd ../snaplabs-website
npm install chainlit-react
```

2. Import and use the component in your layout or pages.

## Deployment

1. Deploy the Python backend to Railway:
   - Connect your GitHub repository
   - Set up the Python environment
   - Add environment variables if needed

2. Update the Chainlit endpoint in your React component to point to the deployed URL

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

MIT
