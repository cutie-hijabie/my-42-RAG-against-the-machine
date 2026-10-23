import os
import tempfile
from pathlib import Path
from collections import Counter
from src.tokenization import tokenize
from typing import Optional, List
from src.models import (
    IndexSettings,
    Passage,
    CorpusFileState,
    IndexData,
    Posting,
)


class Indexer:
    def __init__(
        self,
        settings: Optional[IndexSettings] = None,
        index_path: str = "data/processed",
    ) -> None:
        self.settings = settings if settings is not None else IndexSettings()
        self.index_path = index_path

    def build_index(
        self,
        passages: List[Passage],
        corpus_path: str,
        files: List[CorpusFileState],
    ) -> IndexData:
        if not passages:
            raise ValueError(
                "Cannot build an index from an empty passage list."
            )

        document_lengths = []
        postings = {}

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

    def 
