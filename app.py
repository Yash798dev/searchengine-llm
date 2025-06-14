import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.streamlit import StreamlitCallbackHandler

# Load environment
load_dotenv()

# Streamlit App config
st.set_page_config(page_title="Agentic AI Assistant")
st.title("🔍 AI Search Assistant with LangChain + Tools")

# Sidebar for API key
st.sidebar.title("🔐 Settings")
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything related to AI, research, or current topics."}]

# Show chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Input
if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Chat container for assistant
    response_box = st.chat_message("assistant")

    with response_box:
        # ✅ Callback output container to show internal reasoning
        callback_container = st.container()
        st_cb = StreamlitCallbackHandler(callback_container, expand_new_thoughts=True)

        try:
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name="Llama3-8b-8192",
                streaming=True
            )

            # Tools
            arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2))
            wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
            search_tool = DuckDuckGoSearchRun()

            tools = [wiki_tool, search_tool, arxiv_tool]

            # Initialize Agent
            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False  # Important: disable terminal prints
            )

            # Invoke with callback
            result = agent.invoke({"input": prompt}, callbacks=[st_cb])

            # Final result to UI
            response_box.markdown(f"**Answer:** {result}")
            st.session_state.messages.append({"role": "assistant", "content": result})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
