import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import tempfile
import os

from langchain_groq import ChatGroq
from langchain_classic.agents import initialize_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_classic.agents.agent_types import AgentType
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def graph_generator(df):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        csv_path = tmp.name
        df.to_csv(csv_path, index=False)
    
    llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")
    tool = [PythonREPLTool()]
    custom_prompt = """
    You are a smart AI agent. If you encounter Python code, you must execute it using the Python tool
    Only use tools if required to solve the user's request.
    """
    agent = initialize_agent(
        tools = tool,
        llm = llm,
        agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors = True,
        verbose = True,
        agent_kwargs = {"system_message":custom_prompt}
    )

    column_info = "\n".join([f"{col} ({dtype})"for col,dtype in df.dtypes.items()])

    prompt = f"""
    You are a data visualization assistant.

    The dataset is stored in a file named '{csv_path}'.
    Here are the column names and data types:
    {column_info}

    1. Load it using pandas.
    2. Generate 2 or 3 meaningful and relevant plots using matplotlib or seaborn.
    3. Save each plot using `plt.savefig()` as 'chart1.png', 'chart2.png', etc.
    4. Do not show the plots or add explanations. Just save them.

    Only return valid Python code.
    """
    output = agent.run(prompt)
    images = []
    for i in range(1,4):
        path = f"chart{i}.png"
        if os.path.exists(path):
            images.append(path)
    return images