from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

def demo_init_chat_model():
    chat_model = init_chat_model(
        model="openrouter/free",
        model_provider="openrouter",
        temperature=0.7,
        streaming=True,
        max_retries=3,
    )

    prompt = ChatPromptTemplate.from_template(
        "Name the capital of {place}. One word only"
    )

    parser = StrOutputParser()

    chain = prompt | chat_model | parser

    result = chain.invoke({"place": "France"})

    print(result)

if __name__ == "__main__":
    demo_init_chat_model()