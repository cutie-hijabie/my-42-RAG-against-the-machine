import os
from typing import List
from src.models import SourceFile


def read_source_file(file_path: str) -> SourceFile:
    with open(file_path, "r", encoding="utf-8", newline="") as file:
        file_guts = file.read()

    validated_file_guts = SourceFile(file_path=file_path, text=file_guts)

    return validated_file_guts


def load_sample_corpus(
    corpus_path: str = "data/raw/vllm-0.10.1",
) -> List[SourceFile]:
    result = []

    fake_file1 = "README.md"
    fake_file2 = "find_cuda_init.py"

    fake_path1 = os.path.join(corpus_path, fake_file1)
    fake_path2 = os.path.join(corpus_path, fake_file2)

    fake_source = read_source_file(fake_path1)
    result.append(fake_source)
    fake_source = read_source_file(fake_path2)
    result.append(fake_source)

    return result
