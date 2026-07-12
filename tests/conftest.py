import os

os.environ["VECTOR_STORE_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "mock"
os.environ["EMBEDDING_DIMENSION"] = "16"
os.environ["EMBEDDING_MODEL"] = "mock-embedding"
os.environ["EMBEDDING_VERSION"] = "test-v1"
os.environ["GROQ_API_KEY"] = ""
os.environ["SEARCH_INTERNAL_TOKEN"] = "test-search-token"
os.environ["NLP_SERVICE_TOKEN"] = "test-nlp-service-token"
os.environ["PUBLIC_GLOBAL_SEARCH_ENABLED"] = "true"
os.environ["GLOBAL_SEARCH_MIN_SEMANTIC_SCORE"] = "0.30"
os.environ["GLOBAL_SEARCH_MIN_LEXICAL_SCORE"] = "0.05"
os.environ["GLOBAL_SEARCH_THRESHOLD_POLICY_VERSION"] = "test-search-policy-v1"
