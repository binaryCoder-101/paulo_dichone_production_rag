from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# single text
text = "This is a sample text to be embedded."
embedding = embeddings.embed_query(text)
print(f"Embedding for single text: {embedding}")
