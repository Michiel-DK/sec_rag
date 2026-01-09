from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="sec_rag",
    version="0.1.0",
    description="SEC RAG - Retrieval Augmented Generation for SEC filings",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.12",
)