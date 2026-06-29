import os

os.environ["VECTOR_STORE_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "mock"
os.environ["EMBEDDING_DIMENSION"] = "16"
os.environ["EMBEDDING_MODEL"] = "mock-embedding"
os.environ["EMBEDDING_VERSION"] = "test-v1"
os.environ["GROQ_API_KEY"] = ""
