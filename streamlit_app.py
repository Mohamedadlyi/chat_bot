import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, ClientSettings
from groq import Groq

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

client = Groq(
    api_key=GROQ_API_KEY
)

# Initialize the Whisper model

def get_response(query, model):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model=model,
    )

    return chat_completion.choices[0].message.content

def transcribe_audio(audio_file):
    transcription = client.audio.transcriptions.create(
      file= audio_file,
      model="whisper-large-v3",
      prompt="Specify context or spelling",  # Optional
      response_format="json",  # Optional
      temperature=0.0  # Optional
    )
    return transcription.text


class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.audio_buffer = []

    def recv(self, frame):
        self.audio_buffer.append(frame.to_ndarray())
        return frame

    def get_audio_buffer(self):
        return self.audio_buffer


def main():
    st.title("Chatbot")  # Title at the top

    model_options = {"Gemma 7b":'gemma-7b-it', "Mixtral 8x7b" : "mixtral-8x7b-32768", "LLaMA3 70b": "llama3-70b-8192", "LLaMA3 8b": "llama3-8b-8192"}  # Add your model options here
    selected_model = st.selectbox("Select a model", list(model_options.keys()))  # Dropdown menu below the title
    selected_model_id = model_options[selected_model]

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # File uploader for voice input
    audio_file = st.file_uploader("Upload a voice file", type=["wav", "mp3", "m4a"])

    # Recorder button for direct mic input
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        client_settings=ClientSettings(
            media_stream_constraints={
                "audio": True,
                "video": False,
            }
        ),
    )
    if webrtc_ctx.state.playing:
        if st.button("Stop Recording"):
            webrtc_ctx.state = WebRtcMode.stopped
            audio_buffer = webrtc_ctx.audio_processor.get_audio_buffer()
            if audio_buffer:
                with open("temp_audio.wav", "wb") as f:
                    f.write(b"".join(audio_buffer))
                with st.spinner("Transcribing audio..."):
                    user_input = transcribe_audio("temp_audio.wav")
                    st.text_area("Transcribed Text", value=user_input, height=100)
    elif audio_file is not None:
        with st.spinner("Transcribing audio..."):
            user_input = transcribe_audio(audio_file)
            st.text_area("Transcribed Text", value=user_input, height=100)
    else:
        user_input = st.text_input("Enter your message", key="user_input")  # Text input below the dropdown menu

    if st.button("Send") and user_input:  # Send button below the text input
        response_container = st.empty()
        full_response = ""

        conversation_history = "\n".join([f"You: {msg[0]}\nBot: {msg[1]}" for msg in st.session_state.conversation])
        combined_input = f"{conversation_history}\nYou: {user_input}\nBot:"

        with st.spinner("Wait for bot response..."):
            response = get_response(combined_input, selected_model_id)
            full_response = response
            response_container.text_area("Bot", value=full_response, height=200)  # Bot response below the Send button

        st.session_state.conversation.append((user_input, full_response))

    conversation_text = "\n".join([f"You: {msg[0]}\nBot: {msg[1]}" for msg in st.session_state.conversation])
    st.text_area("History", value=conversation_text, height=400, key="conversation_text_area")  # Conversation history at the bottom

if __name__ == "__main__":
    main()