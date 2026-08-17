from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}.")

messages = prompt.format_messages(adjective="funny", topic="chickens")

print(messages)