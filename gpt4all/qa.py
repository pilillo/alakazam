import os

import streamlit as st
from get_gpt4all import load

from langchain.agents import AgentType, initialize_agent, Tool
from langchain.utilities import WikipediaAPIWrapper
from langchain import LLMMathChain
from langchain.tools import DuckDuckGoSearchRun, BaseTool, StructuredTool, Tool, tool
from pydantic import BaseModel, Field


llm = load(target_folder=os.getcwd(), model_name="ggml-gpt4all-l13b-snoozy.bin", n_ctx=512)

class CalculatorInput(BaseModel):
    question: str = Field()

tools = [
    Tool(
        name = "wikipedia",
        func= WikipediaAPIWrapper().run,
        description="useful for when you need to search for static information on an encyclopedia"
    ),
    Tool(
        name='DuckDuckGo Search',
        func= DuckDuckGoSearchRun().run,
        description="to be used when you need to search up to date information on the internet or that any other tool is not able to find. When using this tool try to summarize the input."
    ),
    Tool.from_function(
        func=LLMMathChain(llm=llm, verbose=True).run,
        name="Calculator",
        description="useful for when you need to answer questions about math",
        args_schema=CalculatorInput
    )
]


agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm, verbose=False, max_iterations=4,
)

page_title="GPT4All QA Agent"
st.set_page_config(page_title=page_title)
st.header(page_title)

user_input = st.text_input('Input your prompt here')

if user_input:
    response = agent.run(user_input)
    st.write(response)