import streamlit as st
import json
from pathlib import Path


# Voice output function - available in all modes
@st.fragment
def create_voice_output_button(text_content, button_key):
    """
    Creates a simple voice output button with pause/resume functionality
    Args:
        text_content (str): The text to convert to speech
        button_key (str): Unique key for the button
    """
    if not text_content or not text_content.strip():
        return

    # Clean text for better speech
    clean_text = (
        text_content.replace("**", "")
        .replace("*", "")
        .replace("`", "")
        .replace("\n", " ")
        .strip()
    )

    # Simple button with pause/resume functionality
    audio_html = f"""
    <button id="voiceBtn_{button_key}" onclick="toggleSpeech_{button_key}()" 
            style="background: transparent; 
                   border: 1px solid white; 
                   border-radius: 8px; 
                   padding: 8px 12px; 
                   cursor: pointer;
                   transition: all 0.2s ease;
                   font-size: 16px;"
            onmouseover="this.style.backgroundColor='rgba(255,255,255,0.1)'"
            onmouseout="this.style.backgroundColor='transparent'">
        <span id="voiceIcon_{button_key}">🔊</span>
    </button>

    <script>
        let utterance_{button_key} = null;
        let isPlaying_{button_key} = false;
        let isPaused_{button_key} = false;
        
        function toggleSpeech_{button_key}() {{
            const icon = document.getElementById('voiceIcon_{button_key}');
            
            if (!window.speechSynthesis) {{
                return;
            }}
            
            // If currently playing, pause
            if (isPlaying_{button_key} && !isPaused_{button_key}) {{
                window.speechSynthesis.pause();
                isPaused_{button_key} = true;
                icon.innerHTML = '▶️';
                return;
            }}
            
            // If paused, resume
            if (isPaused_{button_key}) {{
                window.speechSynthesis.resume();
                isPaused_{button_key} = false;
                icon.innerHTML = '⏸️';
                return;
            }}
            
            // Start new speech
            window.speechSynthesis.cancel();
            utterance_{button_key} = new SpeechSynthesisUtterance(`{clean_text}`);
            
            utterance_{button_key}.onstart = function() {{
                isPlaying_{button_key} = true;
                isPaused_{button_key} = false;
                icon.innerHTML = '⏸️';
            }};
            
            utterance_{button_key}.onend = function() {{
                isPlaying_{button_key} = false;
                isPaused_{button_key} = false;
                icon.innerHTML = '🔊';
            }};
            
            utterance_{button_key}.onerror = function() {{
                isPlaying_{button_key} = false;
                isPaused_{button_key} = false;
                icon.innerHTML = '🔊';
            }};
            
            window.speechSynthesis.speak(utterance_{button_key});
        }}
    </script>
    """

    # Display the simple button
    st.components.v1.html(audio_html, height=40)
    # st.markdown(audio_html, unsafe_allow_html=True)


# Helper function to create inline label with voice button
def create_inline_label_with_voice(label_text, text_content, button_key):
    """
    Creates an inline label with voice button like "Messages: 🔊"
    """
    if not text_content or not text_content.strip():
        st.write(f"**{label_text}**")
        return

    # Clean text for better speech
    clean_text = (
        text_content.replace("**", "")
        .replace("*", "")
        .replace("`", "")
        .replace("\n", " ")
        .strip()
    )

    # Create inline label with voice button
    inline_html = f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
        <span style="color: #888888; 
                     font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif; 
                     font-weight: 600; 
                     font-size: 16px; 
                     letter-spacing: 0.5px;">
            {label_text}
        </span>
        <button id="voiceBtn_{button_key}" onclick="toggleSpeech_{button_key}()" 
                style="background: transparent; 
                       border: 1px solid #888888;
                       border-radius: 6px; 
                       padding: 4px 8px; 
                       cursor: pointer;
                       transition: all 0.2s ease;
                       font-size: 14px;
                       margin: 0;
                       vertical-align: middle;"
                onmouseover="this.style.backgroundColor='rgba(136,136,136,0.1)'"
                onmouseout="this.style.backgroundColor='transparent'">
            <span id="voiceIcon_{button_key}">🔊</span>
        </button>
    </div>

    <script>
        let utterance_{button_key} = null;
        let isPlaying_{button_key} = false;
        let isPaused_{button_key} = false;
        
        function toggleSpeech_{button_key}() {{
            const icon = document.getElementById('voiceIcon_{button_key}');
            
            if (!window.speechSynthesis) {{
                return;
            }}
            
            // If currently playing, pause
            if (isPlaying_{button_key} && !isPaused_{button_key}) {{
                window.speechSynthesis.pause();
                isPaused_{button_key} = true;
                icon.innerHTML = '▶️';
                return;
            }}
            
            // If paused, resume
            if (isPaused_{button_key}) {{
                window.speechSynthesis.resume();
                isPaused_{button_key} = false;
                icon.innerHTML = '⏸️';
                return;
            }}
            
            // Start new speech
            window.speechSynthesis.cancel();
            utterance_{button_key} = new SpeechSynthesisUtterance(`{clean_text}`);
            
            utterance_{button_key}.onstart = function() {{
                isPlaying_{button_key} = true;
                isPaused_{button_key} = false;
                icon.innerHTML = '⏸️';
            }};
            
            utterance_{button_key}.onend = function() {{
                isPlaying_{button_key} = false;
                isPaused_{button_key} = false;
                icon.innerHTML = '🔊';
            }};
            
            utterance_{button_key}.onerror = function() {{
                isPlaying_{button_key} = false;
                isPaused_{button_key} = false;
                icon.innerHTML = '🔊';
            }};
            
            window.speechSynthesis.speak(utterance_{button_key});
        }}
    </script>
    """

    # Display the inline label with voice button
    st.components.v1.html(inline_html, height=50)
    # st.markdown(inline_html, unsafe_allow_html=True)


# Load the config file
config_path = Path("src/settings_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# Conditionally import streamlit_mic_recorder only in debug mode
if config.get("mode") == "debug":
    from streamlit_mic_recorder import speech_to_text


# Helper function to create audio input and callback
def create_audio_input_for_run(key_prefix, text_key):
    if f"{key_prefix}_output" not in st.session_state:
        st.session_state[f"{key_prefix}_output"] = ""

    if text_key not in st.session_state:
        st.session_state[text_key] = ""

    def update_text():
        audio_text = st.session_state.get(f"{key_prefix}_output", "").strip()
        if audio_text:
            st.session_state[text_key] = audio_text
            st.rerun()

    st.session_state[f"update_{key_prefix}"] = update_text

    # Only use speech_to_text if in debug mode
    try:
        mode = config.get("mode")
    except Exception:
        mode = None
    if mode == "debug":
        # Make sure speech_to_text is defined/imported
        _ = speech_to_text(
            language="en",
            start_prompt="🎙️",
            stop_prompt="🔴",
            just_once=False,
            callback=update_text,
            args=(),
            kwargs={},
            key=f"{key_prefix}",
        )
    return st.session_state.get(text_key, "").strip()
