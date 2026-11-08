import os
import tempfile
from pathlib import Path
from collections import Counter
from src.tokenization import tokenize
from typing import Dict, Optional, List
from src.corpus import CorpusLoader
from src.models import (
    IndexSettings,
    Passage,
    CorpusFileState,
    IndexData,
    Posting,
)


class Indexer:
    """Build, save, and load the index used for BM25 retrieval.

    Build a lookup from tokens to the passages where they appear,
    along with the token counts needed for ranking. Save the index
    with its settings and corpus snapshot so it can be validated
    before reuse.

    Attributes:
        settings: Settings used to build the index. Loading an
            existing index updates this to the saved settings.
        index_path: Directory containing the ``index.json`` file.
    """

    def __init__(
        self,
        settings: Optional[IndexSettings] = None,
        index_path: str = "data/processed",
    ) -> None:
        """Initialize the indexer with settings and a storage directory.

        Args:
            settings: Settings to use when building an index. If
                omitted, create an ``IndexSettings`` with defaults.
            index_path: Directory where ``index.json`` is saved and
                loaded. Defaults to ``data/processed``.
        """
        self.settings = settings if settings is not None else IndexSettings()
        self.index_path = index_path

    def build_index(
        self,
        passages: List[Passage],
        corpus_path: str,
        files: List[CorpusFileState],
    ) -> IndexData:
        """Build an in-memory inverted index from source passages.

        Assign each passage a chunk ID based on its position in the
        input list. Tokenize its text and create one posting per
        distinct token in that passage. Each posting records the
        chunk ID and the number of times that token occurs there.

        Record each passage's length in tokens and calculate the
        average passage length for BM25 scoring.

        Args:
            passages: Ordered passages to index.
            corpus_path: Path identifying the source corpus.
            files: Snapshot of the source files used to build the
                index, including their paths, sizes, and modification
                times.

        Returns:
            The in-memory index containing the passages, postings,
            length statistics, settings, and corpus snapshot.

        Raises:
            ValueError: If there are no passages or tokenization
                produces no searchable terms.
            ValidationError: If the resulting index data fails
                model validation.
        """
        if not passages:
            raise ValueError(
                "Cannot build an index from an empty passage list."
            )

        document_lengths = []
        postings: Dict[str, List[Posting]] = {}

        for chunk_id, passage in enumerate(passages):
            passage_tokens = tokenize(passage.text)
            document_lengths.append(len(passage_tokens))
            token_counts = Counter(passage_tokens)

            for token, count in token_counts.items():
                postings.setdefault(token, [])

                token_posting = Posting(
                    chunk_id=chunk_id, term_frequency=count
                )
                postings[token].append(token_posting)

        if not postings:
            raise ValueError(
                "Cannot build an index: no searchable terms were found."
            )

        average_document_length = sum(document_lengths) / len(document_lengths)

        return IndexData(
            format_version=1,
            settings=self.settings,
            corpus_path=corpus_path,
            files=files,
            passages=passages,
            postings=postings,
            document_lengths=document_lengths,
            average_document_length=average_document_length,
        )

    def save_index(self, index: IndexData) -> str:
        """Validate and write an index to ``index.json``.

        Revalidate the index and write its JSON to a temporary file
        in the destination directory. Once the write finishes and
        the temporary file closes, atomically replace ``index.json``
        so readers see a complete file. Attempt to remove the
        temporary file if it remains after the operation.

        Args:
            index: In-memory index to validate and save.

        Returns:
            The path to the saved ``index.json`` file.

        Raises:
            ValidationError: If the index fails revalidation.
            OSError: If the destination directory cannot be created
                or the index file cannot be written or replaced.
        """
        validated_index = IndexData.model_validate(index.model_dump())

        index_directory = Path(self.index_path)
        index_directory.mkdir(parents=True, exist_ok=True)
        final_path = index_directory / "index.json"
        index_json = validated_index.model_dump_json(indent=2)

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", delete=False, dir=index_directory
            ) as temp_file:
                temp_path = temp_file.name
                temp_file.write(index_json)

            os.replace(temp_path, final_path)

        finally:
            if temp_path is not None:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    pass

        return str(final_path)

    def load_index(self) -> IndexData:
        """Load a saved index and check it against the current corpus.

        Read ``index.json`` and validate it as ``IndexData``. Compare
        the saved and current file snapshots by path so that their
        list order does not affect the comparison. If the snapshots
        match, adopt the index's saved settings.

        Returns:
            The validated index with its saved passages and postings.

        Raises:
            OSError: If the index cannot be opened or read, or if
                corpus metadata cannot be collected.
            UnicodeDecodeError: If the index file is not valid UTF-8.
            ValidationError: If the JSON is invalid or fails index
                model validation.
            ValueError: If the saved corpus path fails validation or
                the current snapshot differs from the saved snapshot.
        """
        index_path_obj = Path(self.index_path)
        index_path_obj = index_path_obj / "index.json"

        with open(index_path_obj, encoding="utf-8") as index_file:
            index_json = index_file.read()
        index: IndexData = IndexData.model_validate_json(index_json)

        corpus = CorpusLoader(index.corpus_path)
        corpus_data = corpus.snapshot_files()

        saved_files = {}

        for file in index.files:
            saved_files[file.file_path] = file

        current_files = {}

        for file in corpus_data:
            current_files[file.file_path] = file

        if saved_files != current_files:
            raise ValueError("Corpus files have changed. Rebuild the index.")

        self.settings = index.settings

        return index
