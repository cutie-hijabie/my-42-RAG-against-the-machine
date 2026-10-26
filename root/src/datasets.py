from pydantic import ValidationError
from src.models import RagDataset
from src.validation import require_file


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

