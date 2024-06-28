import streamlit as st
from groq import Groq


GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

client = Groq(
    api_key=GROQ_API_KEY
)

def get_response(query):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        model="llama3-70b-8192",
    )

    return chat_completion.choices[0].message.content

def main():
    st.title("Chatbot")

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    user_input = st.text_input("Enter your message", key="user_input")

    if st.button("Send") and user_input:
        response_container = st.empty()  
        full_response = ""
        
        conversation_history = "\n".join([f"You: {msg[0]}\nBot: {msg[1]}" for msg in st.session_state.conversation])
        combined_input = f"{conversation_history}\nYou: {user_input}\nBot:"

        with st.spinner("Wait for bot response..."):
            for response in get_response(combined_input):
                full_response += response
                response_container.text_area("Bot", value=full_response, height=200)
        
        st.session_state.conversation.append((user_input, full_response))

    conversation_text = "\n".join([f"You: {msg[0]}\nBot: {msg[1]}" for msg in st.session_state.conversation])
    st.text_area("History", value=conversation_text, height=400, key="conversation_text_area")

if __name__ == "__main__":
    main()
