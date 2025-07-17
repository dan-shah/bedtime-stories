import streamlit as st
import anthropic
import os
import openai
import sounddevice as sd
import soundfile as sf
import tempfile
import numpy as np
import base64
import time
import random
from typing import Optional
from dotenv import load_dotenv
from elevenlabs import ElevenLabs

# Load environment variables
load_dotenv()

def retry_with_backoff(func, max_retries=3, base_delay=1, max_delay=60, backoff_factor=2):
    """
    Retry function with exponential backoff for handling API rate limits and temporary errors.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for delay on each retry
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            error_str = str(e)
            
            # Check if this is a retryable error
            if any(error_type in error_str.lower() for error_type in [
                'overloaded', 'rate limit', 'timeout', '429', '500', '502', '503', '504'
            ]):
                if attempt < max_retries:
                    # Calculate delay with jitter
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    jitter = random.uniform(0.1, 0.3) * delay
                    total_delay = delay + jitter
                    
                    st.warning(f"API temporarily overloaded. Retrying in {total_delay:.1f} seconds... (Attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(total_delay)
                    continue
                else:
                    st.error(f"API still overloaded after {max_retries} retries. Please try again later.")
                    raise e
            else:
                # Non-retryable error, raise immediately
                raise e

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

def generate_story_images(client: openai.OpenAI, story: str, story_prompt: str, child_name: str = "", model: str = "dall-e-2") -> tuple[Optional[str], Optional[str]]:
    """Generate two images for the story using DALL-E - one for each half."""
    try:
        # Split story into two halves
        sentences = story.split('. ')
        mid_point = len(sentences) // 2
        first_half = '. '.join(sentences[:mid_point])
        second_half = '. '.join(sentences[mid_point:])
        
        # Create child-friendly, illustration-style prompts with explicit no-text instructions
        base_style = "Children's book illustration style, soft colors, whimsical, gentle, appropriate for bedtime stories. NO TEXT, NO WORDS, NO LETTERS, NO WRITING in the image. Pure illustration only"
        
        # Generate prompts for each half
        first_prompt = f"{base_style}. Visual scene showing: {story_prompt}"
        if child_name:
            first_prompt += f" with a child character"
        first_prompt += f". Beginning scene: {first_half[:150]}... NO TEXT OR WORDS in image"
        
        second_prompt = f"{base_style}. Visual scene showing: {story_prompt}"
        if child_name:
            second_prompt += f" with a child character"
        second_prompt += f". Ending scene: {second_half[:150]}... NO TEXT OR WORDS in image"
        
        # Generate first image with retry logic
        def _generate_first_image():
            # Set quality parameter only for DALL-E 3
            kwargs = {
                "model": model,
                "prompt": first_prompt,
                "size": "1024x1024",
                "n": 1,
            }
            if model == "dall-e-3":
                kwargs["quality"] = "standard"
            
            response = client.images.generate(**kwargs)
            return response.data[0].url
        
        first_image_url = retry_with_backoff(_generate_first_image)
        
        # Generate second image with retry logic
        def _generate_second_image():
            # Set quality parameter only for DALL-E 3
            kwargs = {
                "model": model,
                "prompt": second_prompt,
                "size": "1024x1024",
                "n": 1,
            }
            if model == "dall-e-3":
                kwargs["quality"] = "standard"
            
            response = client.images.generate(**kwargs)
            return response.data[0].url
        
        second_image_url = retry_with_backoff(_generate_second_image)
        
        return first_image_url, second_image_url
        
    except Exception as e:
        st.error(f"Error generating images: {str(e)}")
        return None, None

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

def create_audio_player(audio_bytes: bytes, voice_name: str = "") -> str:
    """Create a simple HTML audio player."""
    b64 = base64.b64encode(audio_bytes).decode()
    
    html = f'''
    <div style="margin: 20px 0;">
        <h4>🎧 {voice_name}</h4>
        <audio controls style="width: 100%;">
            <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
            Your browser does not support the audio element.
        </audio>
    </div>
    '''
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

    def _generate_story():
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.7,
            messages=[{"role": "user", "content": story_prompt}]
        )
        return message.content[0].text
    
    try:
        return retry_with_backoff(_generate_story)
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
                    ("Domi - Cheerful Female", "AZnzlk1XvdvUeBnXmlld"),
                    ("Demon Monster", "vfaqCOvlrKi4Zp7C2IAm")
                ],
                format_func=lambda x: x[0]
            )
            selected_voice_id = voice_option[1]
            
            enable_narration = st.checkbox("🎧 Enable Story Narration", value=True)
        else:
            st.info("💡 Add your ElevenLabs API key to enable voice narration!")
            enable_narration = False
            selected_voice_id = None
        
        # Image generation settings
        if openai_client:
            st.markdown("---")
            st.header("🎨 Image Settings")
            enable_images = st.checkbox("🖼️ Generate Story Illustrations", value=True)
            if enable_images:
                dalle_model = st.selectbox(
                    "Image Model",
                    ["dall-e-2", "dall-e-3"],
                    index=0,
                    help="DALL-E 2 tends to avoid text in images, DALL-E 3 has higher quality but may include unwanted text"
                )
                st.info("💡 Two illustrations will be generated: one for the beginning and one for the end of the story")
            else:
                dalle_model = "dall-e-2"
        else:
            st.info("💡 Add your OpenAI API key to enable story illustrations!")
            enable_images = False
            dalle_model = "dall-e-2"
        
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
                if "story_images" in st.session_state:
                    del st.session_state.story_images
    
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
            
            # Generate story images if enabled
            if enable_images and openai_client:
                with st.spinner("🎨 Creating beautiful illustrations for your story..."):
                    first_image_url, second_image_url = generate_story_images(
                        openai_client, story, prompt, child_name, dalle_model
                    )
                    if first_image_url and second_image_url:
                        st.session_state.story_images = {
                            "first_image": first_image_url,
                            "second_image": second_image_url
                        }
    
    # Display story if it exists
    if "generated_story" in st.session_state:
        story = st.session_state.generated_story
        
        st.markdown("---")
        st.subheader("🌙 Your Bedtime Story")
        
        # Show story text with images integrated if no audio generated yet
        if "story_audio" not in st.session_state or not st.session_state.story_audio:
            if "story_images" in st.session_state:
                # Split story into sentences for better image placement
                sentences = story.split('. ')
                sentences = [s.strip() + '.' if not s.endswith('.') else s.strip() for s in sentences if s.strip()]
                
                # Calculate midpoint for image placement
                mid_point = len(sentences) // 2
                
                # First half of story
                first_half = ' '.join(sentences[:mid_point])
                second_half = ' '.join(sentences[mid_point:])
                
                # Display first half
                st.markdown(first_half)
                
                # Display middle image
                st.markdown("---")
                st.markdown("### 🎨 Illustration")
                st.image(st.session_state.story_images["first_image"], use_column_width=True)
                st.markdown("---")
                
                # Display second half
                st.markdown(second_half)
                
                # Display ending image
                st.markdown("---")
                st.markdown("### 🌙 The End")
                st.image(st.session_state.story_images["second_image"], use_column_width=True)
                st.markdown("---")
            else:
                # No images, show story normally
                st.markdown(story)
        
        # Image generation controls for existing stories
        if openai_client and "story_images" not in st.session_state:
            st.markdown("### 🎨 Image Controls")
            if st.button("🖼️ Generate Illustrations for This Story"):
                metadata = st.session_state.get("story_metadata", {})
                with st.spinner("🎨 Creating beautiful illustrations for your story..."):
                    first_image_url, second_image_url = generate_story_images(
                        openai_client, story, 
                        metadata.get("prompt", ""), 
                        metadata.get("child_name", ""),
                        "dall-e-2"  # Default to DALL-E 2 for manual generation
                    )
                    if first_image_url and second_image_url:
                        st.session_state.story_images = {
                            "first_image": first_image_url,
                            "second_image": second_image_url
                        }
                        st.rerun()
        
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
                
                # Show audio players
                for voice_id, audio_data in st.session_state.story_audio.items():
                    audio_html = create_audio_player(
                        audio_data["audio"], 
                        audio_data['voice_name']
                    )
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
                
                # Interactive highlighting demonstration
                st.subheader("🎯 Interactive Read-Along")
                
                if st.button("▶️ Start Read-Along Demo (3 seconds per sentence)"):
                    # Note: Read-along feature works best without images for now
                    if "story_images" in st.session_state:
                        st.info("💡 Read-along demo works best when images are not displayed. Clear the story and regenerate without images for the full read-along experience.")
                    
                    # Split story into sentences for the demo
                    import re
                    sentences = re.split(r'(?<=[.!?])\s+', story.strip())
                    sentences = [s.strip() for s in sentences if s.strip()]
                    
                    # Create a placeholder for the highlighted story
                    story_placeholder = st.empty()
                    
                    for i, sentence in enumerate(sentences):
                        # Create highlighted version
                        highlighted_story = ""
                        for j, sent in enumerate(sentences):
                            if j < i:
                                # Completed sentences - green background
                                highlighted_story += f'<span style="background-color: #c8e6c9; padding: 2px 4px; border-radius: 3px; opacity: 0.8;">{sent}</span> '
                            elif j == i:
                                # Current sentence - yellow highlight
                                highlighted_story += f'<span style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">{sent}</span> '
                            else:
                                # Future sentences - normal
                                highlighted_story += f'<span style="padding: 2px 4px;">{sent}</span> '
                        
                        # Update the story display
                        story_placeholder.markdown(f'''
                        <div style="
                            margin-top: 20px; 
                            padding: 20px; 
                            background-color: #f8f9fa; 
                            border-radius: 10px;
                            font-size: 1.1em;
                            line-height: 1.8;
                            border-left: 4px solid #007acc;
                        ">
                            {highlighted_story}
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Wait 3 seconds before next sentence
                        time.sleep(3)
                    
                    # Final state - all completed
                    final_story = ""
                    for sent in sentences:
                        final_story += f'<span style="background-color: #c8e6c9; padding: 2px 4px; border-radius: 3px; opacity: 0.8;">{sent}</span> '
                    
                    story_placeholder.markdown(f'''
                    <div style="
                        margin-top: 20px; 
                        padding: 20px; 
                        background-color: #f8f9fa; 
                        border-radius: 10px;
                        font-size: 1.1em;
                        line-height: 1.8;
                        border-left: 4px solid #007acc;
                    ">
                        {final_story}
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    st.success("✅ Read-along complete! Perfect for following along with the audio.")
    
    elif generate_button:
        st.warning("Please enter a story idea first!")

if __name__ == "__main__":
    main()