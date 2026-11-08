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
    """Read and validate a JSON question dataset.

    Decode the file as UTF-8 and validate its contents as a
    ``RagDataset``. Accept annotated and unannotated questions,
    preserve their input order, and require unique question IDs.

    Args:
        dataset_path: Path to the JSON question dataset.

    Returns:
        The validated question dataset.

    Raises:
        FileNotFoundError: If the input file does not exist.
        OSError: If the input file cannot be opened or read.
        UnicodeDecodeError: If the input file is not valid UTF-8.
        ValueError: If the path is not a regular file, the JSON or
            dataset structure is invalid, or question IDs repeat.
    """

    dataset_path = require_file(dataset_path)

    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        dataset = dataset_file.read()
        try:
            datasets: RagDataset = RagDataset.model_validate_json(dataset)
        except ValidationError as error:
            raise ValueError(
                f"Invalid JSON or dataset structure in {dataset_path}: {error}"
            ) from error

    question_ids = set()

    for data in datasets.rag_questions:
        if data.question_id in question_ids:
            raise ValueError(f"Duplicate question ID: {data.question_id}")

        question_ids.add(data.question_id)

    return datasets


def load_search_results(
    student_search_results_path: str,
) -> StudentSearchResults:
    """Read and validate saved retrieval results.

    Decode the file as UTF-8 and validate its contents as
    ``StudentSearchResults``. Require unique question IDs and
    preserve the stored question order, ranked source locations,
    and retrieval limit.

    Args:
        student_search_results_path: Path to the JSON file containing
            saved retrieval results.

    Returns:
        The validated retrieval results.

    Raises:
        FileNotFoundError: If the input file does not exist.
        OSError: If the input file cannot be opened or read.
        UnicodeDecodeError: If the input file is not valid UTF-8.
        ValueError: If the path is not a regular file, the JSON or
            results structure is invalid, or question IDs repeat.
    """
    student_search_results_path = require_file(student_search_results_path)

    with open(
        student_search_results_path, "r", encoding="utf-8"
    ) as student_search_results_file:
        student_search_result = student_search_results_file.read()
        try:
            student_search_results: StudentSearchResults = (
                StudentSearchResults.model_validate_json(student_search_result)
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
    """Validate and save retrieval results or answers as JSON.

    Create the destination directory as needed and revalidate the
    results before serialization. Write indented JSON to a temporary
    file in the destination directory, then atomically replace the
    final file once the write is complete. Attempt to remove any
    temporary file left after the operation.

    Args:
        results: Retrieval results or retrieval results with answers.
        save_directory: Directory where the JSON file is written.
        filename: File basename ending in ``.json`` or ``.JSON``,
            without directory components.

    Returns:
        The path to the saved JSON file.

    Raises:
        ValueError: If the filename is invalid.
        TypeError: If the results are not a supported model type.
        ValidationError: If the results fail model revalidation.
        OSError: If the destination directory cannot be created or
            the output file cannot be written or replaced.
    """
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
            "results must be StudentSearchResults or "
            "StudentSearchResultsAndAnswer."
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
