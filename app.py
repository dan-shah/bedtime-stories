import streamlit as st
import anthropic
import os
import openai
import sounddevice as sd
import soundfile as sf
import tempfile
import numpy as np
import base64
from typing import Optional
from dotenv import load_dotenv
from elevenlabs import ElevenLabs

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="Bedtime Story Generator",
    page_icon="📚",
    layout="wide"
)

def get_claude_client() -> Optional[anthropic.Anthropic]:
    """Initialize Claude client with API key."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)

def get_openai_client() -> Optional[openai.OpenAI]:
    """Initialize OpenAI client with API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return openai.OpenAI(api_key=api_key)

def get_elevenlabs_client() -> Optional[ElevenLabs]:
    """Initialize ElevenLabs client with API key."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return None
    return ElevenLabs(api_key=api_key)

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone."""
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    return recording, sample_rate

def transcribe_audio(client: openai.OpenAI, audio_data, sample_rate):
    """Transcribe audio using OpenAI Whisper."""
    try:
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sample_rate)
            
            # Transcribe using Whisper
            with open(temp_file.name, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            return transcript.text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def text_to_speech(client: ElevenLabs, text: str, voice_id: str = "pNInz6obpgDQGcFmaJgB") -> Optional[bytes]:
    """Convert text to speech using ElevenLabs."""
    try:
        # Generate speech using the correct method
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_monolingual_v1"
        )
        
        # Convert generator to bytes
        audio_bytes = b"".join(audio)
        return audio_bytes
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

def create_audio_player(audio_bytes: bytes) -> str:
    """Create an HTML audio player for the generated speech."""
    b64 = base64.b64encode(audio_bytes).decode()
    html = f"""
    <audio controls autoplay>
        <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
    """
    return html

def generate_story(client: anthropic.Anthropic, prompt: str, child_name: str = "", theme: str = "") -> str:
    """Generate a bedtime story using Claude."""
    
    story_prompt = f"""Create a gentle, imaginative bedtime story for a 6-year-old child. The story should be:
- 3-4 minutes long when read aloud (approximately 400-600 words)
- Age-appropriate with a comforting, peaceful ending
- Creative and engaging but not overstimulating before bedtime
- Include a gentle moral or lesson

Story idea: {prompt}

{f"Main character name: {child_name}" if child_name else ""}
{f"Theme/Setting: {theme}" if theme else ""}

Please write a complete story with a clear beginning, middle, and end. Make it warm and soothing for bedtime."""

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.7,
            messages=[{"role": "user", "content": story_prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error generating story: {str(e)}"

def main():
    st.title("📚 Bedtime Story Generator")
    st.markdown("*Create magical bedtime stories for your little one*")
    
    # Initialize clients
    claude_client = get_claude_client()
    openai_client = get_openai_client()
    elevenlabs_client = get_elevenlabs_client()
    
    if not claude_client:
        st.error("⚠️ Please set your ANTHROPIC_API_KEY environment variable to use this app.")
        st.markdown("You can get an API key from [Anthropic's Console](https://console.anthropic.com/)")
        return
    
    # Sidebar for customization
    with st.sidebar:
        st.header("Story Settings")
        child_name = st.text_input("Child's Name (optional)", placeholder="e.g., Emma")
        theme = st.selectbox(
            "Story Theme",
            ["", "Adventure", "Animals", "Magic", "Space", "Ocean", "Forest", "Castle", "Friendship"]
        )
        
        # Voice settings
        if elevenlabs_client:
            st.markdown("---")
            st.header("🔊 Voice Settings")
            voice_option = st.selectbox(
                "Narrator Voice",
                [
                    ("Burt Reynolds", "4YYIPFl9wE5c4L2eu2Gb"),
                    ("DanShahDotCom", "L2Ztarb5Q7APkwWdQTDy"),
                    ("Adam - Warm Male", "pNInz6obpgDQGcFmaJgB"),
                    ("Bella - Gentle Female", "EXAVITQu4vr4xnSDxMaL"),
                    ("Rachel - Storyteller", "21m00Tcm4TlvDq8ikWAM"),
                    ("Antoni - Deep Male", "ErXwobaYiN019PkySvjV"),
                    ("Domi - Cheerful Female", "AZnzlk1XvdvUeBnXmlld")
                ],
                format_func=lambda x: x[0]
            )
            selected_voice_id = voice_option[1]
            
            enable_narration = st.checkbox("🎧 Enable Story Narration", value=True)
        else:
            st.info("💡 Add your ElevenLabs API key to enable voice narration!")
            enable_narration = False
            selected_voice_id = None
        
        st.markdown("---")
        st.markdown("### How to use:")
        st.markdown("1. Enter a simple story idea")
        st.markdown("2. Customize settings if desired")
        st.markdown("3. Click 'Generate Story'")
        st.markdown("4. Enjoy reading together!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Story Idea")
        
        # Voice input section
        if openai_client:
            st.markdown("**🎤 Voice Input** (Optional)")
            
            if st.button("🎤 Record Story Idea (5 seconds)"):
                with st.spinner("Recording... Speak now!"):
                    recording, sample_rate = record_audio(duration=5)
                    sd.wait()  # Wait for recording to finish
                
                with st.spinner("Transcribing your voice..."):
                    transcription = transcribe_audio(openai_client, recording, sample_rate)
                    if not transcription.startswith("Error"):
                        st.session_state.prompt = transcription
                        st.success(f"Heard: {transcription}")
                    else:
                        st.error(transcription)
            
            if st.button("🎤 Record Longer Idea (10 seconds)"):
                with st.spinner("Recording... Speak now!"):
                    recording, sample_rate = record_audio(duration=10)
                    sd.wait()  # Wait for recording to finish
                
                with st.spinner("Transcribing your voice..."):
                    transcription = transcribe_audio(openai_client, recording, sample_rate)
                    if not transcription.startswith("Error"):
                        st.session_state.prompt = transcription
                        st.success(f"Heard: {transcription}")
                    else:
                        st.error(transcription)
        else:
            st.info("💡 Add your OpenAI API key to the .env file to enable voice input!")
        
        st.markdown("**✍️ Text Input**")
        prompt = st.text_area(
            "What kind of story would you like? (1-3 sentences)",
            value=st.session_state.get("prompt", ""),
            placeholder="A brave little mouse who discovers a secret door in the library...",
            height=100
        )
        
        generate_button = st.button("✨ Generate Story")
        
        if "generated_story" in st.session_state:
            if st.button("🗑️ Clear Story"):
                if "generated_story" in st.session_state:
                    del st.session_state.generated_story
                if "story_audio" in st.session_state:
                    del st.session_state.story_audio
                if "story_metadata" in st.session_state:
                    del st.session_state.story_metadata
    
    with col2:
        st.subheader("Story Examples")
        examples = [
            "A sleepy dragon who forgot how to breathe fire",
            "A star that fell from the sky and needed help getting home",
            "A little girl who could talk to her stuffed animals",
            "A magical tree that grew different fruits for different wishes"
        ]
        
        for example in examples:
            if st.button(f"📖 {example}", key=example):
                st.session_state.prompt = example
    
    # Generate and display story
    if generate_button and prompt:
        with st.spinner("Creating your magical story..."):
            story = generate_story(claude_client, prompt, child_name, theme)
            st.session_state.generated_story = story
            st.session_state.story_metadata = {
                "prompt": prompt,
                "child_name": child_name,
                "theme": theme
            }
    
    # Display story if it exists
    if "generated_story" in st.session_state:
        story = st.session_state.generated_story
        
        st.markdown("---")
        st.subheader("🌙 Your Bedtime Story")
        st.markdown(story)
        
        # Voice controls section
        if elevenlabs_client:
            st.markdown("### 🎤 Voice Controls")
            
            if st.button("🎧 Generate Audio with Selected Voice"):
                with st.spinner(f"🎤 Generating voice narration with {voice_option[0]}..."):
                    audio_bytes = text_to_speech(elevenlabs_client, story, selected_voice_id)
                    
                    if audio_bytes:
                        # Store audio in session state with voice name
                        if "story_audio" not in st.session_state:
                            st.session_state.story_audio = {}
                        st.session_state.story_audio[selected_voice_id] = {
                            "audio": audio_bytes,
                            "voice_name": voice_option[0]
                        }
            
            # Display all generated audio versions
            if "story_audio" in st.session_state and st.session_state.story_audio:
                st.markdown("### 🔊 Available Audio Versions")
                for voice_id, audio_data in st.session_state.story_audio.items():
                    st.markdown(f"**{audio_data['voice_name']}**")
                    audio_html = create_audio_player(audio_data["audio"])
                    st.markdown(audio_html, unsafe_allow_html=True)
                    
                    # Download button for each voice
                    st.download_button(
                        label=f"💾 Download {audio_data['voice_name']} Version",
                        data=audio_data["audio"],
                        file_name=f"bedtime_story_{audio_data['voice_name'].replace(' ', '_')}.mp3",
                        mime="audio/mpeg",
                        key=f"download_{voice_id}"
                    )
                    st.markdown("---")
            
            # Add some styling to make it feel more story-like
            st.markdown("""
            <style>
            .story-text {
                font-size: 1.1em;
                line-height: 1.6;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 10px;
                margin: 20px 0;
            }
            </style>
            """, unsafe_allow_html=True)
    
    elif generate_button:
        st.warning("Please enter a story idea first!")

if __name__ == "__main__":
    main()