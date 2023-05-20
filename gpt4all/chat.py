import os

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import streamlit as st
from streamlit_chat import message
from get_gpt4all import load

llm = load(target_folder=os.getcwd(), model_name="ggml-gpt4all-l13b-snoozy.bin")
chain = ConversationChain(llm=llm, verbose=False, memory=ConversationBufferMemory())

def generate_response(user_input):
    return chain.predict(input=user_input)

page_title="GPT4All Chat"
st.set_page_config(page_title=page_title)
st.header(page_title)

input_container = st.container()
response_container = st.container()

# list of generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm HugChat, How may I help you?"]

# user's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

user_input = st.text_input("Input your prompt here", key="input")

with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))