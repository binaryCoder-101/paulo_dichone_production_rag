from dotenv import load_dotenv
from importlib.metadata import version

load_dotenv()

core_version = version("langchain_core")
lg_version = version("langgraph")
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_openrouter import ChatOpenRouter

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")

def main():
    print("Hello from paulo-dicole-langchain-projects!")

    # llm_openai = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # response_openai = llm_openai.invoke("Say 'setup complete' in one word")
    # print(f"Response from ChatOpenAI: {response_openai}")

    # llm_anthropic = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    # response_anthropic = llm_anthropic.invoke("Say 'setup complete' in one word")
    # print(f"Response from ChatAnthropic: {response_anthropic}")

    llm_openrouter = ChatOpenRouter(model="openrouter/free", temperature=0)
    response_openrouter = llm_openrouter.invoke("Say 'setup complete' in one word")
    print(f"Response from ChatOpenRouter: {response_openrouter}")

if __name__ == "__main__":
    main()
