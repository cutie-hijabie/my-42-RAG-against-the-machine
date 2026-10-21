import os
from tqdm import tqdm
from typing import List
from pathlib import Path
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


def raise_walk_error(error: OSError) -> None:
    raise error


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

    file_paths = []

    for root, dirs, files in os.walk(
        corpus_path_obj,
        topdown=True,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        dirs_we_want = []

        for directory in dirs:
            dir_path = Path(root) / directory

            if dir_path.is_symlink():
                continue

            if str(directory).startswith("."):
                continue

            dirs_we_want.append(directory)

        dirs[:] = dirs_we_want

        for filename in files:
            file_path = Path(root) / filename

            if file_path.is_symlink():
                continue

            if not str(file_path).endswith((".py", ".md", ".txt", ".rst")):
                continue

            if not file_path.is_file():
                continue

            file_path = file_path.resolve()

            if not file_path.is_relative_to(project_root):
                raise ValueError(
                    "Source file must be inside the project root."
                )

            relative_path = file_path.relative_to(project_root).as_posix()

            file_paths.append(relative_path)

    return sorted(file_paths)


def load_corpus(
    corpus_path: str = "data/raw/vllm-0.10.1",
) -> List[SourceFile]:

    source_files = discover_source_files(corpus_path)

    if not source_files:
        raise ValueError(
            "No supported source files were found in the corpus directory."
        )

    source_file_records = []

    for source_file in tqdm(source_files, desc="Loading corpus", unit="file"):
        source_file = read_source_file(source_file)
        source_file_records.append(source_file)

    return source_file_records
