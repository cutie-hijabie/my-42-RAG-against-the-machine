import os
import tempfile
from pydantic import ValidationError
from src.models import (
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.validation import require_file
from pathlib import Path


def load_dataset(dataset_path: str) -> RagDataset:

    dataset_path = require_file(dataset_path)

    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        dataset = dataset_file.read()
        try:
            dataset = RagDataset.model_validate_json(dataset)
        except ValidationError as error:
            raise ValueError(
                f"Invalid JSON or dataset structure in {dataset_path}: {error}"
            ) from error

    question_ids = set()

    for data in dataset.rag_questions:
        if data.question_id in question_ids:
            raise ValueError(f"Duplicate question ID: {data.question_id}")

        question_ids.add(data.question_id)

    return dataset


def load_search_results(
    student_search_results_path: str,
) -> StudentSearchResults:
    student_search_results_path = require_file(student_search_results_path)

    with open(
        student_search_results_path, "r", encoding="utf-8"
    ) as student_search_results_file:
        student_search_results = student_search_results_file.read()
        try:
            student_search_results = StudentSearchResults.model_validate_json(
                student_search_results
            )
        except ValidationError as error:
            raise ValueError(
                f"Invalid JSON or search-results structure in "
                f"{student_search_results_path}: {error}"
            ) from error

    question_ids = set()

    for result in student_search_results.search_results:
        if result.question_id in question_ids:
            raise ValueError(f"Duplicate question ID: {result.question_id}")

        question_ids.add(result.question_id)

    return student_search_results


def save_results(
    results: StudentSearchResults | StudentSearchResultsAndAnswer,
    save_directory: str,
    filename: str,
) -> str:
    if (
        not filename.endswith((".json", ".JSON"))
        or "/" in filename
        or "\\" in filename
        or not filename
    ):
        raise ValueError(
            "filename must be a .json basename with no directory components."
        )

    save_directory_path = Path(save_directory)
    save_directory_path.mkdir(parents=True, exist_ok=True)

    if isinstance(results, StudentSearchResults):
        validated_results = StudentSearchResults.model_validate(
            results.model_dump()
        )
    elif isinstance(results, StudentSearchResultsAndAnswer):
        validated_results = StudentSearchResultsAndAnswer.model_validate(
            results.model_dump()
        )
    else:
        raise TypeError(
            "results must be StudentSearchResults or StudentSearchResultsAndAnswer."
        )

    results_json = validated_results.model_dump_json(indent=2)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=save_directory
        ) as temp_file:
            temp_path = temp_file.name
            temp_file.write(results_json)

        os.replace(temp_path, save_directory_path / filename)

    finally:
        if temp_path is not None:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass

    return str(save_directory_path / filename)
