from src.corpus import CorpusLoader
from src.chunking import Chunker
from src.indexing import Indexer
from tqdm import tqdm
from time import perf_counter
import sys
from typing import NoReturn
from src.models import (
    IndexSettings,
    UnansweredQuestion,
    MinimalSearchResults,
    StudentSearchResults,
)
from src.retrieval import Retriever
from src.datasets import save_results


class Cli:
    def index(
        self,
        max_chunk_size: int = 2000,
        corpus_path: str = "data/raw/vllm-0.10.1",
        index_path: str = "data/processed",
        overlap: int = 150,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> str:

        start_time = perf_counter()
        try:
            settings = IndexSettings(
                max_chunk_size=max_chunk_size,
                overlap=overlap,
                k1=k1,
                b=b,
            )
            loader = CorpusLoader(corpus_path)
            chunker = Chunker(settings)
            indexer = Indexer(settings=settings, index_path=index_path)
            before = loader.snapshot_files()
            source_files = loader.load_corpus()
            passages = []

            for source in tqdm(
                source_files, desc="Chunking corpus", unit="file"
            ):
                passages.extend(chunker.chunk_source(source))

            after = loader.snapshot_files()

            if before != after:
                raise ValueError(
                    "Corpus files changed during indexing. Run indexing again."
                )

            index = indexer.build_index(
                passages,
                loader.corpus_path,
                after,
            )

            if loader.snapshot_files() != after:
                raise ValueError(
                    "Corpus files changed while building the index. "
                    "Run indexing again."
                )
            saved_path = indexer.save_index(index)

            elapsed = perf_counter() - start_time
            print(
                f"Indexing completed in {elapsed:.2f} seconds.",
                file=sys.stderr,
            )

            return saved_path
        except (ValueError, OSError, UnicodeError) as error:
            self.fail(error)

    def fail(self, error: Exception) -> NoReturn:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)

    def search(
        self,
        query: str,
        k: int = 5,
        index_path: str = "data/processed",
        save_directory: str = "data/output/search_results/single",
    ) -> str:
        try:
            question = UnansweredQuestion(question=query)
            indexer = Indexer(index_path=index_path)
            index = indexer.load_index()
            retriever = Retriever(index)
            sources = retriever.search(question.question, k)
            result = MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=sources,
            )
            results = StudentSearchResults(
                search_results=[result],
                k=k,
            )
            saved_path = save_results(
                results,
                save_directory=save_directory,
                filename="query.json",
            )
            return saved_path
        except (ValueError, OSError, UnicodeError) as error:
            self.fail(error)
