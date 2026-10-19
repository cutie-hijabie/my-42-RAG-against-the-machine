class Cli:

    def index(self, max_chunk_size: int = 2000,
          corpus_path: str = "data/raw/vllm-0.10.1",
          index_path: str = "data/processed",
          overlap: int = 150, k1: float = 1.5,
          b: float = 0.75) -> str:

    def search(self, query: str, k: int = 5) -> str:
        return f"{query} + {k}"
