from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}.")

messages = prompt.format_messages(adjective="funny", topic="chickens")

print(messages)


# Multi-message template

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        (
            "human",
            "Translate the following text: {text}",
        ),
    ]
)

messages = prompt.format_messages(
    input_language="English", output_language="French", text="I love programming (for the money ofcourse!)"
)

print(messages)

# Message Types:

from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

messages = [
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi there! How can I assist you today?"),
    SystemMessage(content="This is a system message."),
    ToolMessage(content="Tool executed successfully.", tool_call_id="call_123"),
    ChatMessage(role="user", content="This is a general chat message."),
]

print(messages)