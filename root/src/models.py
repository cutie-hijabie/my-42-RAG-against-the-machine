from uuid import uuid4
from math import isclose
from typing import List, Dict, Annotated
from .validation import validate_k, validate_chunk_size
from pydantic import BaseModel, Field, field_validator, model_validator


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int = Field(ge=0, strict=True)
    last_character_index: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def validate_source_range(self) -> "MinimalSource":
        if self.last_character_index <= self.first_character_index:
            raise ValueError("Source ranges must satisfy 0 <= start < end.")

        return self


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

    @field_validator("k", mode="before")
    @classmethod
    def validate_result_k(cls, value: int) -> int:
        return validate_k(value)


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int

    @field_validator("k", mode="before")
    @classmethod
    def validate_result_k(cls, value: int) -> int:
        return validate_k(value)


class SourceFile(BaseModel):
    file_path: str
    text: str


class Passage(MinimalSource):
    text: str

    @model_validator(mode="after")
    def validate_text_length(self) -> "Passage":
        space_btween = self.last_character_index - self.first_character_index
        if space_btween != len(self.text):
            raise ValueError(
                "Passage text length must equal last_character_index "
                "minus first_character_index."
            )
        return self


class IndexSettings(BaseModel):
    max_chunk_size: int = 2000
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

    @field_validator("max_chunk_size", mode="before")
    @classmethod
    def validate_settings_chunk_size(cls, value: int) -> int:
        return validate_chunk_size(value)


class CorpusFileState(BaseModel):
    file_path: str
    size_bytes: int = Field(ge=0, strict=True)
    mtime_ns: int = Field(ge=0, strict=True)


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
    document_lengths: List[Annotated[int, Field(ge=0, strict=True)]]
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

        file_paths = {file.file_path for file in self.files}

        if len(file_paths) != len(self.files):
            raise ValueError(
                "Index file metadata must contain unique file paths."
            )

        for passage in self.passages:
            if passage.file_path not in file_paths:
                raise ValueError(
                    "Every passage must reference a file listed "
                    "in index metadata."
                )

            if len(passage.text) > self.settings.max_chunk_size:
                raise ValueError(
                    "Passage length must not exceed settings.max_chunk_size."
                )

        return self


class PreparedContext(BaseModel):
    prompt_text: str
    included_sources: List[MinimalSource]
    input_token_count: int = Field(ge=0, strict=True)
    has_evidence: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_evidence_consistency(self) -> "PreparedContext":
        if self.has_evidence != bool(self.included_sources):
            raise ValueError(
                "has_evidence must match whether included_sources is nonempty."
            )
        return self


class QuestionRecall(BaseModel):
    question_id: str
    found_sources: int = Field(ge=0, strict=True)
    expected_sources: int = Field(ge=0, strict=True)
    recall: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_recall_consistency(self) -> "QuestionRecall":
        if self.found_sources > self.expected_sources:
            raise ValueError("found_sources must not exceed expected_sources.")

        if self.expected_sources > 0:
            expected_recall = self.found_sources / self.expected_sources
        else:
            expected_recall = 0.0

        if not isclose(
            self.recall,
            expected_recall,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "recall must match found_sources / expected_sources, "
                "or be zero when expected_sources is zero."
            )

        return self


class EvaluationReport(BaseModel):
    k: int = Field(ge=0, strict=True)
    question_scores: List[QuestionRecall]
    mean_recall: float = Field(ge=0, le=1, allow_inf_nan=False)
    missing_question_ids: List[str]
    skipped_question_ids: List[str]

    @model_validator(mode="after")
    def validate_report_consistency(self) -> "EvaluationReport":
        question_ids = [score.question_id for score in self.question_scores]

        if len(question_ids) != len(set(question_ids)):
            raise ValueError(
                "question_scores must contain unique question IDs."
            )

        if len(self.missing_question_ids) != len(
            set(self.missing_question_ids)
        ):
            raise ValueError(
                "missing_question_ids must not contain duplicates."
            )

        if len(self.skipped_question_ids) != len(
            set(self.skipped_question_ids)
        ):
            raise ValueError(
                "skipped_question_ids must not contain duplicates."
            )

        evaluable_scores = [
            score
            for score in self.question_scores
            if score.expected_sources > 0
        ]

        if not evaluable_scores:
            raise ValueError(
                "Evaluation report must contain at least one evaluable question."
            )

        expected_mean = sum(score.recall for score in evaluable_scores) / len(
            evaluable_scores
        )

        if not isclose(
            self.mean_recall,
            expected_mean,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "mean_recall must match the mean recall "
                "of evaluable questions."
            )

        return self
