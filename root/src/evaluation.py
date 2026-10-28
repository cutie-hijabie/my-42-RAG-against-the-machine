from src.models import (
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    EvaluationReport,
    AnsweredQuestion,
)
from src.validation import validate_k
from math import isfinite


class Evaluator:
    def __init__(
        self, k: int | None = None, iou_threshold: float = 0.5
    ) -> None:
        if k is not None:
            k = validate_k(k)

        if (
            isinstance(iou_threshold, bool)
            or not isinstance(iou_threshold, float)
            or not isfinite(iou_threshold)
        ):
            raise ValueError(
                "iou_threshold must be a finite number greater than 0 and "
                "less than or equal to 1; booleans are not allowed."
            )

        self.k = k
        self.iou_threshold = iou_threshold

    def source_iou(self, a: MinimalSource, b: MinimalSource) -> float:
        if a.file_path != b.file_path:
            return 0.0
        overlap_start = max(a.first_character_index, b.first_character_index)
        overlap_end = min(a.last_character_index, b.last_character_index)
        intersection = max(0, overlap_end - overlap_start)

        length_a = a.last_character_index - a.first_character_index
        length_b = b.last_character_index - b.first_character_index
        union = length_a + length_b - intersection

        return intersection / union

    def evaluate_results(
        self, student: StudentSearchResults, reference: RagDataset
    ) -> EvaluationReport:

        k = self.k if self.k is not None else student.k

        if k > student.k:
            raise ValueError(
                "Evaluation k must not exceed the saved search results' k."
            )

        reference_questions = {}

        for question in reference.rag_questions:
            if question.question_id in reference_questions:
                raise ValueError(
                    f"Duplicate reference question ID: {question.question_id}"
                )
            if not isinstance(question, AnsweredQuestion):
                raise TypeError(
                    f"Reference question must be annotated: {question.question_id}"
                )

            reference_questions[question.question_id] = question

        for result in student.search_results:
            reference_question = reference_questions[result.question_id]

            if result.question != reference_question.question:
                raise ValueError(
                    f"Question text mismatch for ID: {result.question_id}"
                )

        question_scores = []
        missing_question_ids = []
        skipped_question_ids = []

        for question in reference.rag_questions:
            result = student_results.get(question.question_id)

            if result is None:
                missing_question_ids.append(question.question_id)
                predictions = []
            else:
                predictions = result.retrieved_sources[:k]
