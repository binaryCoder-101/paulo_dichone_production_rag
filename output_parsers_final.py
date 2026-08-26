from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(model="gpt-4o-mini", temperature=0)

def demo_str_parser():
    prompt = ChatPromptTemplate.from_template(
        "Explain briefly: {topic}"
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"topic": "SHA-256"})

    print(f"Result: '{result}' (type: {type(result).__name__})")

def demo_json_parser():
    """JSON output parser."""

    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with keys 'city' and 'country' for: {place}\n"
        "Return ONLY valid JSON, no explanation."
    )
    parser = JsonOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"place": "The Eiffel Tower"})
    print(f"Result: '{result}' (type: {type(result).__name__})")
    print(f"City: {result['city']}, Country: {result['country']}")

def demo_pydantic_parser():
    class Recipe(BaseModel):
        name: str = Field(description="Name of the recipe")
        ingredients: List[str] = Field(description="List of ingredients")
        prep_time_minutes: int = Field(description="Preparation time in minutes")
        difficulty: str = Field(description="easy, medium or hard")

    parser = PydanticOutputParser(pydantic_object=Recipe)

    prompt = ChatPromptTemplate.from_template(
        "Create a simple recipe for: {dish}\n\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser

    result = chain.invoke({"dish": "prime rib"})
    print(f"Recipe: {result.name}")
    print(f"Ingredients: {result.ingredients}")
    print(f"Prep time: {result.prep_time_minutes} mins")
    print(f"Difficulty: {result.difficulty}")

    # Type-Checking (Recommended)
    print(
        f"\nType check - name is str: {isinstance(result.name, str)}"
    )
    print(
        f"Type check - ingredients is list: {isinstance(result.ingredients, list)}"
    )
    print(
        f"Type check - every ingredient is str: {all(isinstance(item, str) for item in result.ingredients)}"
    )
    print(
        f"\nType check - prep_time is int: {isinstance(result.prep_time_minutes, int)}"
    )
    print(
        f"\nType check - difficulty is str: {isinstance(result.difficulty, str)}"
    )

if __name__ == "__main__":
    # print("=" * 50)
    # print("Demo 1: String Parser")
    # print("=" * 50)
    # demo_str_parser()

    # print("\n" + "=" * 50)
    # print("Demo 2: JSON Parser")
    # print("=" * 50)
    # demo_json_parser()

    print("\n" + "=" * 50)
    print("Demo 3: Pydantic Parser")
    print("=" * 50)
    demo_pydantic_parser()

    # print("\n" + "=" * 50)
    # print("Demo 4: Structured Output (Modern)")
    # print("=" * 50)
    # demo_structured_output()

    # print("\n" + "=" * 50)
    # print("Demo 5: Complex Schema")
    # print("=" * 50)
    # demo_complex_schema()

    # print("\n" + "=" * 50)
    # print("Exercise: Movie Extraction")
    # print("=" * 50)
    # exercise_structured_extraction()