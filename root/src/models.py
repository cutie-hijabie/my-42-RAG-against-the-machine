from uuid import uuid4
from math import isclose
from typing import List, Dict, Annotated
from .validation import validate_k, validate_chunk_size
from pydantic import BaseModel, Field, field_validator, model_validator


class MinimalSource(BaseModel):
    """Identify a passage by its source path and character range.

    The range is half-open. It includes the start offset and
    excludes the end offset, following Python's slicing convention.

    Attributes:
        file_path: Path of the source file.
        first_character_index: Inclusive character offset.
        last_character_index: Exclusive character offset.
    """

    file_path: str
    first_character_index: int = Field(ge=0, strict=True)
    last_character_index: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def validate_source_range(self) -> "MinimalSource":
        """Ensure that the end offset comes after the start offset.

        Returns:
            The source record with a valid, nonempty range.

        Raises:
            ValueError: If the end offset is at or before the start.
        """
        if self.last_character_index <= self.first_character_index:
            raise ValueError("Source ranges must satisfy 0 <= start < end.")

        return self


class UnansweredQuestion(BaseModel):
    """Represent a question that can be submitted for retrieval.

    Attributes:
        question_id: Identifier supplied by the caller or generated
            as a UUID string when omitted.
        question: Question text to use as the retrieval query.
    """

    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question annotated with reference evidence.

    Attributes:
        sources: Reference source ranges that support the answer.
        answer: Reference answer text.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent a collection of questions loaded from a dataset.

    Attributes:
        rag_questions: Questions in dataset order. Each question may
            be unannotated or include a reference answer and sources.
    """

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Record the ranked retrieval results for one question.

    Attributes:
        question_id: Identifier of the question that was searched.
        question: Original question text.
        retrieved_sources: Retrieved source ranges in ranked order.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Extend a question's retrieval results with a generated answer.

    Attributes:
        answer: Answer text produced by the generation stage.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Group retrieval results and their requested retrieval limit.

    Attributes:
        search_results: Per-question results containing ranked
            source ranges.
        k: Requested maximum number of sources per question.
    """

    search_results: List[MinimalSearchResults]
    k: int

    @field_validator("k", mode="before")
    @classmethod
    def validate_result_k(cls, value: int) -> int:
        """Validate the retrieval limit before field coercion.

        Args:
            value: Raw value supplied for the retrieval limit.

        Returns:
            The original nonnegative integer, including zero.

        Raises:
            ValueError: If the value is negative, a boolean, or
                another non-integer value.
        """
        return validate_k(value)


class StudentSearchResultsAndAnswer(BaseModel):
    """Group generated answers with their retrieval results.

    Attributes:
        search_results: Per-question records containing retrieved
            sources and generated answers.
        k: Requested maximum number of sources per question.
    """

    search_results: List[MinimalAnswer]
    k: int

    @field_validator("k", mode="before")
    @classmethod
    def validate_result_k(cls, value: int) -> int:
        """Validate the retrieval limit before field coercion.

        Args:
            value: Raw value supplied for the retrieval limit.

        Returns:
            The original nonnegative integer, including zero.

        Raises:
            ValueError: If the value is negative, a boolean, or
                another non-integer value.
        """
        return validate_k(value)


class SourceFile(BaseModel):
    """Represent a source file after it has been read.

    Attributes:
        file_path: Path identifying the source file.
        text: Complete decoded text of the file.
    """

    file_path: str
    text: str


class Passage(MinimalSource):
    """Pair a source range with the passage text used for indexing.

    The text length must equal the distance between the inherited
    start and end offsets.

    Attributes:
        text: Text of the passage identified by the source range.
    """

    text: str

    @model_validator(mode="after")
    def validate_text_length(self) -> "Passage":
        """Check that the passage text has the length of its range.

        Returns:
            The passage after its text length has been checked.

        Raises:
            ValueError: If the text length differs from the end
                offset minus the start offset.
        """
        space_btween = self.last_character_index - self.first_character_index
        if space_btween != len(self.text):
            raise ValueError(
                "Passage text length must equal last_character_index "
                "minus first_character_index."
            )
        return self


class IndexSettings(BaseModel):
    """Store the settings used for chunking and BM25 retrieval.

    Attributes:
        max_chunk_size: Maximum passage length in characters.
            Defaults to 2000.
        overlap: Requested character overlap between split
            passages. Defaults to 150 and is capped at one less
            than ``max_chunk_size``.
        k1: Positive BM25 term-frequency saturation parameter.
            Defaults to 1.5.
        b: BM25 passage-length normalization parameter in the
            interval [0, 1]. Defaults to 0.75.
        tokenizer_version: Tokenizer identifier stored with the
            index. The supported value is ``lexical-v1``.
    """

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
        """Check that the tokenizer version is supported.

        Args:
            value: Tokenizer identifier to validate.

        Returns:
            The supported identifier, ``lexical-v1``.

        Raises:
            ValueError: If the identifier is unsupported.
        """
        if value != "lexical-v1":
            raise ValueError(
                "Unsupported tokenizer_version; expected 'lexical-v1'."
            )
        return value

    @model_validator(mode="after")
    def clamp_overlap(self) -> "IndexSettings":
        """Cap the overlap so it stays below the passage size limit.

        Reduce the overlap to ``max_chunk_size - 1`` when necessary.
        This keeps the configured overlap smaller than a full
        size-limited passage.

        Returns:
            The settings instance with the adjusted overlap.
        """
        self.overlap = min(self.overlap, self.max_chunk_size - 1)

        return self

    @field_validator("max_chunk_size", mode="before")
    @classmethod
    def validate_settings_chunk_size(cls, value: int) -> int:
        """Validate the character limit before field coercion.

        Args:
            value: Raw value supplied for the maximum passage size.

        Returns:
            The original integer in the range 1 through 2000.

        Raises:
            ValueError: If the value is outside the allowed range,
                a boolean, or another non-integer value.
        """
        return validate_chunk_size(value)


class CorpusFileState(BaseModel):
    """Record the metadata used to compare corpus snapshots.

    Attributes:
        file_path: Path identifying the source file.
        size_bytes: File size in bytes at the time of the snapshot.
        mtime_ns: File modification time in nanoseconds at the time
            of the snapshot.
    """

    file_path: str
    size_bytes: int = Field(ge=0, strict=True)
    mtime_ns: int = Field(ge=0, strict=True)


class Posting(BaseModel):
    """Record a token's occurrence count in one indexed passage.

    The token is the key of the posting list that contains this
    record.

    Attributes:
        chunk_id: Zero-based position of the passage in the index.
        term_frequency: Positive number of times the token occurs
            in that passage.
    """

    chunk_id: int = Field(ge=0, strict=True)
    term_frequency: int = Field(gt=0, strict=True)


class IndexData(BaseModel):
    """Store the passages and statistics needed for retrieval.

    Keep the inverted index, passage lengths, and corpus snapshot
    together so the saved index can be validated before reuse.

    Attributes:
        format_version: Version of the stored index format. The
            supported version is 1.
        settings: Chunking and retrieval settings saved with the
            index.
        corpus_path: Path identifying the indexed corpus directory.
        files: Snapshot of the source files used to build the index.
        passages: Indexed passages in chunk ID order.
        postings: Mapping from tokens to lists of passages and term
            frequencies.
        document_lengths: Token count for each passage, in the same
            order as ``passages``.
        average_document_length: Mean passage length in tokens.
    """

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
        """Validate relationships between the stored index fields.

        Require a supported format, at least one passage, and at
        least one searchable term. Check that passage lengths align
        with the passage list and that each term has a nonempty
        posting list with unique, valid chunk IDs.

        Verify that the stored average matches the mean token
        count within floating-point tolerance. Check that file
        metadata paths are unique and that each passage references
        a listed file and fits the configured character limit.

        Returns:
            The index after its consistency checks pass.

        Raises:
            ValueError: If the format or any relationship between
                the index fields is invalid.
        """
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
                "Index must contain at least one searchable "
                "term with a posting."
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
                        "Each term's posting list must"
                        " contain unique chunk IDs."
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
    """Represent the prompt and evidence prepared for generation.

    Attributes:
        prompt_text: Formatted prompt to send to the model.
        included_sources: Source ranges included as prompt evidence.
        input_token_count: Number of input tokens in the prompt.
        has_evidence: Whether the prompt includes any source ranges.
    """

    prompt_text: str
    included_sources: List[MinimalSource]
    input_token_count: int = Field(ge=0, strict=True)
    has_evidence: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_evidence_consistency(self) -> "PreparedContext":
        """Check that the evidence flag matches the source list.

        Returns:
            The prepared context after the flag has been checked.

        Raises:
            ValueError: If ``has_evidence`` differs from whether
                ``included_sources`` is nonempty.
        """
        if self.has_evidence != bool(self.included_sources):
            raise ValueError(
                "has_evidence must match whether included_sources is nonempty."
            )
        return self


class QuestionRecall(BaseModel):
    """Record source retrieval recall for one question.

    Attributes:
        question_id: Identifier of the evaluated question.
        found_sources: Number of reference sources found.
        expected_sources: Total number of reference sources.
        recall: Fraction of reference sources found, or zero when
            the question has no reference sources.
    """

    question_id: str
    found_sources: int = Field(ge=0, strict=True)
    expected_sources: int = Field(ge=0, strict=True)
    recall: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_recall_consistency(self) -> "QuestionRecall":
        """Check that the recall score agrees with its counts.

        Require the found count to be at most the expected count.
        Compare the stored recall with the ratio of those counts
        within floating-point tolerance. Use zero as the expected
        score when the reference count is zero.

        Returns:
            The question score after its consistency checks pass.

        Raises:
            ValueError: If the counts or the recall score disagree.
        """
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
    """Summarize retrieval recall across a set of questions.

    Attributes:
        k: Retrieval cutoff used for evaluation.
        question_scores: Individual question recall scores.
        mean_recall: Arithmetic mean of recall across questions
            with at least one expected source.
        missing_question_ids: IDs without a corresponding
            retrieval result.
        skipped_question_ids: IDs excluded from recall scoring.
    """

    k: int = Field(ge=0, strict=True)
    question_scores: List[QuestionRecall]
    mean_recall: float = Field(ge=0, le=1, allow_inf_nan=False)
    missing_question_ids: List[str]
    skipped_question_ids: List[str]

    @model_validator(mode="after")
    def validate_report_consistency(self) -> "EvaluationReport":
        """Check question ID uniqueness and aggregate recall.

        Require unique IDs among the question scores and within
        each of the missing and skipped ID lists. Calculate the
        mean from scores with at least one expected source and
        compare it with the stored value within floating-point
        tolerance.

        Returns:
            The report after its consistency checks pass.

        Raises:
            ValueError: If any checked list has duplicate IDs, no
                score is evaluable, or the stored mean disagrees
                with the calculated mean.
        """
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
                "Evaluation report must contain at"
                " least one evaluable question."
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
