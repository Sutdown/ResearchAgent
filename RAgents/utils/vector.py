
class VectorMemory:
    def __init__(self, persist_directory: str = "./vector_memory", embedding_model: str = "all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model