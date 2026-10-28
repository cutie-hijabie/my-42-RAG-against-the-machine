from src.models import (
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    EvaluationReport,
    AnsweredQuestion,
    QuestionRecall,
)
from src.validation import validate_k
from math import isfinite


class Evaluator:
    def __init__(
        self, k: int | None = None, iou_threshold: float = 0.05
    ) -> None:
        if k is not None:
            k = validate_k(k)

        if (
            isinstance(iou_threshold, bool)
            or not isinstance(iou_threshold, (int, float))
            or not isfinite(iou_threshold)
            or not 0 < iou_threshold <= 1
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

        student_results = {}

        for result in student.search_results:
            if result.question_id in student_results:
                raise ValueError(
                    f"Duplicate student question ID: {result.question_id}"
                )

            if result.question_id not in reference_questions:
                raise ValueError(
                    f"Unknown student question ID: {result.question_id}"
                )

            reference_question = reference_questions[result.question_id]

            if result.question != reference_question.question:
                raise ValueError(
                    f"Question text mismatch for ID: {result.question_id}"
                )

            student_results[result.question_id] = result

        question_scores = []
        missing_question_ids = []
        skipped_question_ids = []

        for question in reference_questions.values():
            result = student_results.get(question.question_id)

            if result is None:
                missing_question_ids.append(question.question_id)
                predictions = []
            else:
                predictions = result.retrieved_sources[:k]

            expected_sources = []
            seen_locations = set()

            for source in question.sources:
                location = (
                    source.file_path,
                    source.first_character_index,
                    source.last_character_index,
                )

                if location not in seen_locations:
                    seen_locations.add(location)
                    expected_sources.append(source)

            if not expected_sources:
                skipped_question_ids.append(question.question_id)
                continue

            found_sources = 0

            for expected_source in expected_sources:
                for prediction in predictions:
                    if (
                        self.source_iou(expected_source, prediction)
                        >= self.iou_threshold
                    ):
                        found_sources += 1
                        break

            recall = found_sources / len(expected_sources)

            question_scores.append(
                QuestionRecall(
                    question_id=question.question_id,
                    found_sources=found_sources,
                    expected_sources=len(expected_sources),
                    recall=recall,
                )
            )

        if not question_scores:
            raise ValueError(
                "Evaluation requires at least one question with reference sources."
            )

        mean_recall = sum(score.recall for score in question_scores) / len(
            question_scores
        )

        return EvaluationReport(
            k=k,
            question_scores=question_scores,
            mean_recall=mean_recall,
            missing_question_ids=missing_question_ids,
            skipped_question_ids=skipped_question_ids,
        )
