from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Sample documents for testing
SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()

SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''


def recursive_splitter():
    """
    Demonstrates RecursiveCharacterTextSplitter (the default and recommended general-purpose splitter).
    
    Key Notes:
    - Splits text hierarchically using a list of separators: ["\n\n", "\n", " ", ""].
    - First tries to split by double newlines (paragraphs). If chunks are still too large,
      it falls back to single newlines (lines), then spaces (words), then individual characters.
    - Preserves semantic units (keeps paragraphs and sentences together as much as possible).
    - `chunk_size=500`: Maximum character length per chunk.
    - `chunk_overlap=50`: Number of characters shared between adjacent chunks to maintain context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"Original length: {len(SAMPLE_TEXT)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk sizes: {[len(c) for c in chunks]}")
    print(f"\nFirst chunk preview:\n{chunks[0][:200]}...")
    print(f"\nSecond chunk preview:\n{chunks[1][:200]}...")


def chunk_size_comparison():
    """
    Compares the impact of different chunk sizes (200, 500, 1000 chars) on chunk count.

    Key Notes:
    - Small chunks (e.g. 200 chars): High retrieval granularity and precision, but risk
      losing broader surrounding context or cutting sentences in awkward places.
    - Large chunks (e.g. 1000 chars): Retain complete ideas and rich context, but may
      contain irrelevant noise and consume more LLM context window tokens.
    - Standard Rule of Thumb: 500-1000 characters (or ~250-500 tokens) with 10-20% overlap
      is often the sweet spot for general RAG applications.
    """
    sizes = [200, 500, 1000]

    print("=== Chunk Size Comparison ===")
    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=size // 5
        )  # 20% overlap
        chunks = splitter.split_text(SAMPLE_TEXT)
        print(f" Size {size}: {len(chunks)} chunks")


def overlap_demo():
    """
    Demonstrates why chunk overlap is crucial in text splitting.

    Key Notes:
    - Without overlap (`chunk_overlap=0`): Facts or sentences spanning across the chunk boundary
      can be truncated, causing the retriever or LLM to miss critical connection points.
    - With overlap (`chunk_overlap=20`): The end of Chunk 1 and beginning of Chunk 2 share common
      text. This sliding window preserves semantic continuity and prevents broken context.
    """
    text = "The quick brown fox jumps over the lazy dog. " * 10  # Repeated text

    # without overlap
    no_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)

    # with overlap
    with_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)

    chunks_no_overlap = no_overlap.split_text(text)
    chunks_with_overlap = with_overlap.split_text(text)

    print("Without overlap:")
    print(f"  Chunk 1 end: ...{chunks_no_overlap[0][-20:]}")
    print(f"  Chunk 2 start: {chunks_no_overlap[1][:20]}...")

    print("\nWith overlap:")
    print(f"  Chunk 1 end: ...{chunks_with_overlap[0][-20:]}")
    print(f"  Chunk 2 start: {chunks_with_overlap[1][:20]}...")


def markdown_splitter():
    """
    Demonstrates structure-aware splitting using MarkdownHeaderTextSplitter.

    Key Notes:
    - Rather than splitting strictly by character count, it splits based on document headings
      (# H1, ## H2, ### H3).
    - Preserves semantic hierarchy: The heading hierarchy is stripped from the chunk text
      and placed directly into document `metadata` (e.g. `{'h1': ..., 'h2': ...}`).
    - Allows downstream vector stores to perform metadata filtering and helps the LLM
      know exactly which section or topic a chunk belongs to.
    """
    headers_to_consider = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_consider)
    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"Markdown Splitter produced {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i} ---")
        print(f" Metadata: {chunk.metadata}\n")
        print(f" Content: {chunk.page_content[:200]}...\n")


def code_splitter():
    """
    Demonstrates syntax-aware splitting for programming source code.

    Key Notes:
    - Uses `RecursiveCharacterTextSplitter.from_language(Language.PYTHON, ...)`.
    - Instead of splitting blindly on arbitrary characters or newlines, it understands Python's
      grammar (separating on `class`, `def`, control flow blocks, etc.).
    - Prevents broken code blocks, orphaned signatures, or severed functions in technical RAG indices.
    """
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=500, chunk_overlap=50
    )
    chunks = python_splitter.split_text(SAMPLE_CODE)
    print(f"Code Splitter produced {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(chunk[:150] + "..." if len(chunk) > 150 else chunk)


def document_splitter():
    """
    Demonstrates splitting LangChain Document objects loaded from an external file (PDF).

    Key Notes:
    - `split_text()` takes a raw Python string and returns a list of strings (`list[str]`).
    - `split_documents()` takes a list of `Document` objects and returns `list[Document]`.
    - Crucial difference: `split_documents()` preserves and propagates metadata (e.g. `source`,
      `page`, `file_name`) from the original document to every individual chunk.
    - In production RAG, preserving metadata across chunks is mandatory for accurate source citations.
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document

    loader = PyPDFLoader("./documents/langchain_demo.pdf")
    docs = loader.load()

    print(f"Loaded {len(docs)} documents from PDF.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # split the docs
    split_docs = splitter.split_documents(docs)

    print(f"Split into {len(split_docs)} chunks")
    print(f"\nFirst chunk metadata: {split_docs[0].metadata}")
    print(f"First chunk content: {split_docs[0].page_content[:200]}...")
    print(f"\nLast chunk metadata: {split_docs[-1].metadata}")

if __name__ == "__main__":
    # print("=== Recursive Character Text Splitter ===")
    # recursive_splitter()
    # chunk_size_comparison()
    # overlap_demo()
    # print("=== Markdown Header Text Splitter ===")
    # markdown_splitter()
    # print("=== Code Splitter ===")
    # code_splitter()
    print("=== Document Splitter from PDF ===")
    document_splitter()
