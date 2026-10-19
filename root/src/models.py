from uuid import uuid4
from typing import List, Dict
from pydantic import BaseModel, Field, field_validator, model_validator


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int


class SourceFile(BaseModel):
    file_path: str
    text: str


class Passage(MinimalSource):
    text: str


class IndexSettings(BaseModel):
    max_chunk_size: int = Field(default=2000, ge=1, le=2000, strict=True)
    overlap: int = Field(default=150, ge=0, strict=True)
    k1: float = Field(default=1.5, gt=0, allow_inf_nan=False, strict=True)
    b: float = Field(
        default=0.75, ge=0, le=1, allow_inf_nan=False, strict=True
    )
    tokenizer_version: str = "lexical-v1"

    @field_validator("tokenizer_version")
    @classmethod
    def validate_tokenizer_version(cls, value: str) -> str:
        if value != "lexical-v1":
            raise ValueError(
                "Unsupported tokenizer_version; expected 'lexical-v1'."
            )
        return value

    @model_validator(mode="after")
    def clamp_overlap(self) -> "IndexSettings":
        self.overlap = min(self.overlap, self.max_chunk_size - 1)

        return self


class CorpusFileState(BaseModel):
    file_path: str
    size_bytes: int = Field(ge=0, strict=True)
    mtime_ns: int = Field(strict=True)


class Posting(BaseModel):
    chunk_id: int = Field(ge=0, strict=True)
    term_frequency: int = Field(gt=0, strict=True)


class IndexData(BaseModel):
    format_version: int = Field(default=1, strict=True)
    settings: IndexSettings
    corpus_path: str
    files: List[CorpusFileState]
    passages: List[Passage]
    postings: Dict[str, List[Posting]]
    document_lengths: List[int]
    average_document_length: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_index_consistency(self) -> "IndexData":
        if self.format_version != 1:
            raise ValueError(
                "Unsupported format_version; expected 1. Rebuild the index."
            )

        if not self.passages:
            raise ValueError("Index must contain at least one passage.")

        if len(self.document_lengths) != len(self.passages):
            raise ValueError(
                "document_lengths must contain exactly one entry per passage."
            )

        if any(length < 0 for length in self.document_lengths):
            raise ValueError("Document lengths must be nonnegative.")

        if not self.postings:
            raise ValueError(
                "Index must contain at least one searchable term with a posting."
            )

        for term, posting_list in self.postings.items():
            if not term:
                raise ValueError("Index term keys must not be empty.")

            if not posting_list:
                raise ValueError(
                    "Each index term must have at least one posting."
                )

            seen_ids: set[int] = set()

            for posting in posting_list:
                if posting.chunk_id >= len(self.passages):
                    raise ValueError(
                        "Posting chunk_id must reference an existing passage."
                    )

                if posting.chunk_id in seen_ids:
                    raise ValueError(
                        "Each term's posting list must contain unique chunk IDs."
                    )

                seen_ids.add(posting.chunk_id)

        expected_average = sum(self.document_lengths) / len(self.passages)

        if not isclose(
            self.average_document_length,
            expected_average,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "average_document_length must match the mean "
                "of document_lengths."
            )
