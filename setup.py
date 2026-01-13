from setuptools import setup, find_packages

# Minimal core dependencies (not all from requirements.txt)
core_requirements = [
    "langchain>=0.3.0",
    "langchain-chroma>=0.2.0",
    "langchain-community>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-google-genai>=2.0.0",
    "chromadb>=0.6.0",
    "google-genai>=1.14.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
]

setup(
    name="sec_rag",
    version="0.1.0",
    description="SEC RAG - Retrieval Augmented Generation for SEC filings",
    packages=find_packages(),
    install_requires=core_requirements,
    python_requires=">=3.12",
)