from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

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

def demo_model_comparison():
    prompt = "Explain recursion in one sentence"

    models = {
        "gpt-40-mini": init_chat_model(
            model="gpt-4o-mini",
            temperature=0.7,
            streaming=True,
        ),
        "gpt-4o": init_chat_model(
            model="gpt-4o",
            temperature=0.7,
            streaming=True,
        ),
    }

    if os.getenv("ANTHROPIC_API_KEY"):
        models["claude-sonnet-4-5-20250929"] = init_chat_model(
            model="claude-sonnet-4-5-20250929",
            temperature=0.7,
            streaming=True,
        )

    print(f"Prompt: {prompt}\n")

    for model_name, model in models.items():
        response = model.invoke(prompt)
        print(f"Response from {model_name}: {response.content}\n\n")

def demo_multiturn():
    model = init_chat_model(
        model="claude-sonnet-4-5-20250929",
        temperature=0.7,
        streaming=True,
    )

    messages = [
        SystemMessage(content="You are a pirate. Always answer like one!"),
        HumanMessage(content="What's the weather like today?")
    ]

    response = model.invoke(messages)
    print(f"Response from the Pirate: {response.content}")

    messages.append(response)
    messages.append(HumanMessage(content="What about tomorrow?"))

    response = model.invoke(messages)
    print(f"Response from the Pirate: {response.content}")

    print(messages)

if __name__ == "__main__":
    # demo_init_chat_model()
    # demo_model_comparison()
    demo_multiturn()