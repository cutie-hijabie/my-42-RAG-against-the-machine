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
    """Expose the RAG pipeline as commands through Python Fire.

    Coordinate the components that build an index, retrieve sources,
    generate answers, and evaluate retrieval results. Each command
    prepares its inputs, delegates work to the relevant components, and
    handles expected errors at the command-line boundary.
    """

    def index(
        self,
        max_chunk_size: int = 2000,
        corpus_path: str = "data/raw/vllm-0.10.1",
        index_path: str = "data/processed",
        overlap: int = 150,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> str:
        """Build and save a searchable index for the source corpus.

        Load the corpus, split its files into passages, and build an index
        containing term postings and passage lengths for BM25 retrieval.
        Compare corpus snapshots before loading, after chunking, and
        after index construction to detect changes before saving. Report
        the total indexing time to standard error.

        Args:
            max_chunk_size: Maximum passage length in characters.
            corpus_path: Directory containing the source files.
            index_path: Directory where ``index.json`` will be saved.
            overlap: Requested overlap in characters when splitting a
                section into multiple passages.
            k1: BM25 parameter that controls term-frequency saturation.
            b: BM25 parameter that controls passage-length normalization.

        Returns:
            The path to the saved ``index.json`` file.

        Raises:
            SystemExit: If the settings or corpus are invalid, the corpus
                changes during indexing, or an expected file error occurs.
        """

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
        """Report a command failure and exit with status code 2.

        Write the error message to standard error with an ``Error:``
        prefix.

        Args:
            error: Exception whose message should be displayed.

        Raises:
            SystemExit: Always raised with exit status 2.
        """
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)

    def search(
        self,
        query: str,
        k: int = 5,
        index_path: str = "data/processed",
        save_directory: str = "data/output/search_results/single",
    ) -> str:
        """Retrieve and display ranked passages for a single query.

        Create a question record and load the saved index. Use the
        retriever to find up to ``k`` source locations, then read the
        corresponding passage text from the corpus and print it to
        standard error. Save the question, its ID, the ranked source
        locations, and the requested ``k`` in ``query.json``.

        Args:
            query: Question text to search for.
            k: Maximum number of source locations to retrieve.
            index_path: Directory containing the saved ``index.json``.
            save_directory: Directory where ``query.json`` is written.

        Returns:
            The path to the saved ``query.json`` file.

        Raises:
            SystemExit: With status code 2 if validation fails or
                an expected file or decoding error occurs.
        """
        try:
            question = UnansweredQuestion(question=query)
            indexer = Indexer(index_path=index_path)
            index = indexer.load_index()

            retriever = Retriever(index)
            sources = retriever.search(question.question, k)
            passages = reconstruct_passages(
                sources, CorpusLoader(index.corpus_path)
            )

            for number, passage in enumerate(passages, start=1):
                print(
                    f"\n--- Result {number}: {passage.file_path} "
                    f"[{passage.first_character_index}:"
                    f"{passage.last_character_index}] ---",
                    file=sys.stderr,
                )
                print(passage.text, file=sys.stderr)
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
        """Retrieve ranked source locations for every dataset question.

        Validate the retrieval limit and load the question dataset.
        Reuse one loaded index and retriever for all questions while
        displaying a progress bar. Preserve question IDs and input
        order in the saved results.

        Save the results under the dataset's original filename and
        report the total elapsed time to standard error.

        Args:
            dataset_path: Path to the JSON question dataset.
            k: Maximum number of source locations per question.
            save_directory: Directory where the results are written.
            index_path: Directory containing the saved ``index.json``.

        Returns:
            The path to the saved retrieval results file.

        Raises:
            SystemExit: With status code 2 if validation fails or
                an expected file or decoding error occurs.
        """

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
        """Evaluate saved retrieval results against reference sources.

        Load the student search results and an annotated question
        dataset. Evaluate the top ``k`` results for each question
        using a source overlap threshold of 0.05. Return the report
        as indented JSON for the CLI to display.

        Args:
            student_search_results_path: Path to the JSON file
                containing saved retrieval results.
            dataset_path: Path to the annotated reference dataset.
            k: Number of ranked results to consider per question.
                If omitted, use the limit stored in the results.

        Returns:
            A JSON string containing the evaluation report.

        Raises:
            SystemExit: With status code 2 if input validation or
                evaluation fails, a reference question is unannotated,
                or an expected file or decoding error occurs.
        """
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

            json_report: str = eval_report.model_dump_json(indent=2)

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
        """Retrieve evidence and generate an answer for one question.

        Load the saved index and retrieve up to ``k`` source
        locations. Reconstruct the passage text and prepare a prompt
        within the model's token budget. Print the generated answer
        and its generation time, then save the question, answer,
        and retrieved source locations in ``query.json``.

        Args:
            query: Question to answer.
            k: Maximum number of source locations to retrieve.
            index_path: Directory containing the saved index.
            save_directory: Directory where ``query.json`` is written.
            model_name: Name or path of the generation model.
            device: Device selection, either ``auto``, ``cpu``,
                or ``cuda``.
            context_limit: Total token budget for input and output.
            max_new_tokens: Maximum number of tokens to generate,
                reserved from the total context budget.

        Returns:
            The path to the saved answer file.

        Raises:
            SystemExit: With status code 2 if validation fails or
                an expected file, decoding, or generation error
                occurs.
        """

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
                generation_start = perf_counter()
                answer = generator.generate_answer(context)
                generation_elapsed = perf_counter() - generation_start

                print(
                    f"\nAnswer ({generation_elapsed:.2f} seconds)\n{answer}\n",
                    flush=True,
                )

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
        """Generate answers using the sources in saved search results.

        Load the saved retrieval results and reuse one corpus loader
        and generator across all questions. Reconstruct the retrieved
        passages for each question and prepare its prompt within the
        model's token budget.

        Preserve question IDs, input order, retrieved source
        locations, and the saved retrieval limit. Save the answers
        under the input filename and report the total elapsed time
        to standard error. An empty input produces an empty results
        file without initializing the generator.

        Args:
            student_search_results_path: Path to the JSON file
                containing saved retrieval results.
            save_directory: Directory where the answers are written.
            model_name: Name or path of the generation model.
            device: Device selection, either ``auto``, ``cpu``,
                or ``cuda``.
            context_limit: Total token budget for input and output.
            max_new_tokens: Maximum number of tokens to generate
                per answer, reserved from the total context budget.

        Returns:
            The path to the saved dataset answers file.

        Raises:
            SystemExit: With status code 2 if validation fails or
                an expected file, decoding, or generation error
                occurs.
        """

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
