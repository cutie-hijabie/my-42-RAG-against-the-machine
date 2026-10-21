import os
from pathlib import Path
from typing import List
from src.models import SourceFile
from src.validation import require_directory, require_file


def read_source_file(file_path: str) -> SourceFile:

    project_root = Path(__file__).resolve().parent.parent
    source_path = Path(file_path)

    if not source_path.is_absolute():
        source_path = project_root / source_path

    source_path = source_path.resolve()

    if not source_path.is_relative_to(project_root):
        raise ValueError("Source file must be inside the project root.")

    require_file(str(source_path))

    relative_path = source_path.relative_to(project_root).as_posix()

    with open(source_path, "r", encoding="utf-8", newline="") as file:
        file_guts = file.read()

    validated_file_guts = SourceFile(file_path=relative_path, text=file_guts)

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


def discover_source_files(corpus_path: str) -> List[str]:

    project_root = Path(__file__).resolve().parent.parent
    corpus_path_obj = Path(corpus_path)

    if not corpus_path_obj.is_absolute():
        corpus_path_obj = project_root / corpus_path_obj

    if corpus_path_obj.is_symlink():
        raise ValueError("Corpus path must not contain symbolic links.")

    for parent in corpus_path_obj.parents:
        if parent.is_symlink():
            raise ValueError("Corpus path must not contain symbolic links.")

    corpus_path_obj = corpus_path_obj.resolve()

    if not corpus_path_obj.is_relative_to(project_root):
        raise ValueError("Source file must be inside the project root.")

    require_directory(str(corpus_path_obj))

    file_paths=[]

    for root, dirs, files in os.walk(corpus_path_obj, topdown=True, followlinks=False):
        if file.is_symlink():
            continue
        if not 


