import sys
from pathlib import Path
from time import perf_counter
from typing import NoReturn

from tqdm import tqdm

from src.chunking import Chunker
from src.context import prepare_context, reconstruct_passages
from src.corpus import CorpusLoader
from src.datasets import load_dataset, load_search_results, save_results
from src.evaluation import Evaluator
from src.generation import Generator
from src.indexing import Indexer
from src.models import (
    IndexSettings,
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)
from src.retrieval import Retriever
from src.validation import validate_k


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

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results/UnansweredQuestions",
        index_path: str = "data/processed",
    ) -> str:

        start_time = perf_counter()
        try:
            k = validate_k(k)
            dataset = load_dataset(dataset_path)

            indexer = Indexer(index_path=index_path)
            index = indexer.load_index()
            retriever = Retriever(index)

            records = []

            for data in tqdm(
                dataset.rag_questions,
                desc="Searching dataset",
                unit="question",
            ):
                source_result = retriever.search(data.question, k)
                search_result = MinimalSearchResults(
                    question_id=data.question_id,
                    question=data.question,
                    retrieved_sources=source_result,
                )
                records.append(search_result)

            result = StudentSearchResults(search_results=records, k=k)

            filename = Path(dataset_path).name
            saved_path = save_results(result, save_directory, filename)

            elapsed = perf_counter() - start_time
            print(
                f"Dataset search completed in {elapsed:.2f} seconds.",
                file=sys.stderr,
            )
            return saved_path

        except (ValueError, OSError, UnicodeError) as error:
            self.fail(error)

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int | None = None,
    ) -> str:
        try:
            search_results = load_search_results(student_search_results_path)
            datasets = load_dataset(dataset_path)

            evaluator = Evaluator(k=k, iou_threshold=0.05)

            try:
                eval_report = evaluator.evaluate_results(
                    search_results, datasets
                )
            except TypeError as error:
                if str(error).startswith(
                    "Reference question must be annotated:"
                ):
                    self.fail(error)
                raise

            json_report = eval_report.model_dump_json(indent=2)

            return json_report
        except (ValueError, OSError, UnicodeError) as error:
            self.fail(error)

    def answer(
        self,
        query: str,
        k: int = 5,
        index_path: str = "data/processed",
        save_directory: str = "data/output/search_results_and_answer/single",
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "auto",
        context_limit: int = 4096,
        max_new_tokens: int = 512,
    ) -> str:

        try:
            unanswered_questions = UnansweredQuestion(question=query)
            indexer = Indexer(index_path=index_path)
            index = indexer.load_index()
            retriever = Retriever(index)
            result = retriever.search(unanswered_questions.question, k)

            loader = CorpusLoader(index.corpus_path)

            try:
                generator = Generator(
                    model_name=model_name,
                    device=device,
                    context_limit=context_limit,
                    max_new_tokens=max_new_tokens,
                )

                passages = reconstruct_passages(result, loader)
                context = prepare_context(
                    unanswered_questions.question, passages, generator
                )
                answer = generator.generate_answer(context)

            except RuntimeError as error:
                self.fail(error)

            record = MinimalAnswer(
                question_id=unanswered_questions.question_id,
                question=unanswered_questions.question,
                retrieved_sources=result,
                answer=answer,
            )
            results = StudentSearchResultsAndAnswer(
                search_results=[record],
                k=k,
            )
            saved_path = save_results(
                results,
                save_directory=save_directory,
                filename="query.json",
            )

            return saved_path
            record = MinimalAnswer(
                question_id=unanswered_questions.question_id,
                question=unanswered_questions.question,
                retrieved_sources=result,
                answer=answer,
            )
            results = StudentSearchResultsAndAnswer(
                search_results=[record],
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

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = (
            "data/output/search_results_and_answer/UnansweredQuestions"
        ),
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "auto",
        context_limit: int = 4096,
        max_new_tokens: int = 512,
    ) -> str:

        start_time = perf_counter()
        try:
            search_result = load_search_results(student_search_results_path)

            records = []
            if search_result.search_results:
                loader = CorpusLoader()

                try:
                    generator = Generator(
                        model_name=model_name,
                        device=device,
                        context_limit=context_limit,
                        max_new_tokens=max_new_tokens,
                    )

                    for result in tqdm(
                        search_result.search_results,
                        desc="Searching answers",
                        unit="answer",
                    ):
                        passages = reconstruct_passages(
                            result.retrieved_sources, loader
                        )
                        context = prepare_context(
                            result.question, passages, generator
                        )
                        answer = generator.generate_answer(context)

                        record = MinimalAnswer(
                            question_id=result.question_id,
                            question=result.question,
                            retrieved_sources=result.retrieved_sources,
                            answer=answer,
                        )
                        records.append(record)

                except RuntimeError as error:
                    self.fail(error)

            results = StudentSearchResultsAndAnswer(
                search_results=records,
                k=search_result.k,
            )

            filename = Path(student_search_results_path).name
            saved_path = save_results(
                results,
                save_directory=save_directory,
                filename=filename,
            )

            elapsed = perf_counter() - start_time
            print(
                f"Dataset answering completed in {elapsed:.2f} seconds.",
                file=sys.stderr,
            )
            return saved_path

        except (ValueError, OSError, UnicodeError) as error:
            self.fail(error)
