from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

def demo_basic_chain():
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence: {question}"
    )
    model = ChatOpenRouter(model="openrouter/free", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"question": "What is LangChain"})
    print(f"Response: {result}")

    return chain

def demo_batch_execution():
    prompt = ChatPromptTemplate.from_template(
        "Tranlste to French: {text}"
    )
    model = ChatOpenRouter(model="openrouter/free", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    inputs = [
        {"text": "Hello, how are you?"},
        {"text": "What is your name?"},
        {"text": "Where is the nearest restaurant?"},
    ]
    results = chain.batch(inputs)

    for text in zip(inputs, results):
        print(f"Input: {text[0]['text']} => Output: {text[1]}")

def demo_streaming():
    prompt = ChatPromptTemplate.from_template(
            "Write a haiku about: {topic}"
        )
    model = ChatOpenRouter(model="openrouter/free", temperature=0.7)
    parser = StrOutputParser()
    
    chain = prompt | model | parser

    print("Streaming output: ")
    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)
    print()

if __name__ == "__main__":
    # demo_basic_chain()
    # demo_batch_execution()
    demo_streaming()