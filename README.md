# 🌙 Bedtime Stories Generator

A Streamlit application that generates personalized bedtime stories with AI-powered narration and illustrations.

## Features

- **AI Story Generation**: Custom bedtime stories using Claude AI
- **Voice Narration**: Text-to-speech with ElevenLabs voices
- **Story Illustrations**: DALL-E generated images for visual storytelling
- **Interactive Elements**: Read-along demos and customizable themes
- **Audio Recording**: Voice input for story prompts

## 🐳 Docker Deployment

### Prerequisites

- Docker and Docker Compose installed
- API keys for:
  - Anthropic Claude API
  - OpenAI API (for images)
  - ElevenLabs API (for voice)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dan-shah/bedtime-stories.git
   cd bedtime-stories
   ```

2. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
   ```

3. **Run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

4. **Access the application**:
   Open your browser to `http://localhost:8501`

### Alternative Docker Commands

**Build and run manually**:
```bash
# Build the image
docker build -t bedtime-stories .

# Run the container
docker run -p 8501:8501 --env-file .env bedtime-stories
```

**Stop the application**:
```bash
docker-compose down
```

## 🛠️ Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   Create a `.env` file with your API keys

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📝 Usage

1. Enter a story prompt or use voice recording
2. Customize with child's name and theme
3. Generate story with optional illustrations
4. Listen to AI narration with selected voice
5. Enjoy interactive read-along features

## 🔧 Configuration

- **Image Generation**: Choose between DALL-E 2 or DALL-E 3
- **Voice Selection**: Multiple ElevenLabs voice options
- **Story Themes**: Adventure, friendship, magic, and more

## 🚀 Production Deployment

For production use, consider:
- Using a reverse proxy (nginx)
- Adding SSL/TLS termination
- Implementing proper logging
- Setting up monitoring and health checks
- Using managed services for APIs

## 📄 License

This project is open source and available under the MIT License.