import os
from tqdm import tqdm
from typing import List
from pathlib import Path
from src.models import SourceFile, CorpusFileState
from src.validation import require_directory, require_file


def raise_walk_error(error: OSError) -> None:
    raise error


class CorpusLoader:
    def __init__(self, corpus_path: str = "data/raw/vllm-0.10.1") -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        corpus_path_obj = Path(corpus_path)

        if not corpus_path_obj.is_absolute():
            corpus_path_obj = self.project_root / corpus_path_obj

        if corpus_path_obj.is_symlink():
            raise ValueError("Corpus path must not contain symbolic links.")

        for parent in corpus_path_obj.parents:
            if parent.is_symlink():
                raise ValueError(
                    "Corpus path must not contain symbolic links."
                )

        corpus_path_obj = corpus_path_obj.resolve()

        if not corpus_path_obj.is_relative_to(self.project_root):
            raise ValueError("Corpus must be inside the project root.")

        require_directory(str(corpus_path_obj))

        self.corpus_path = corpus_path_obj.relative_to(
            self.project_root
        ).as_posix()

    def read_source_file(self, file_path: str) -> SourceFile:

        source_path = Path(file_path)

        if not source_path.is_absolute():
            source_path = self.project_root / source_path

        source_path = source_path.resolve()

        if not source_path.is_relative_to(self.project_root):
            raise ValueError("Source file must be inside the project root.")

        require_file(str(source_path))

        relative_path = source_path.relative_to(self.project_root).as_posix()

        with open(source_path, "r", encoding="utf-8", newline="") as file:
            file_guts = file.read()

        validated_file_guts = SourceFile(
            file_path=relative_path, text=file_guts
        )

        return validated_file_guts

    def discover_source_files(self) -> List[str]:

        corpus_path_obj = self.project_root / self.corpus_path

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

                if not file_path.is_relative_to(corpus_path_obj):
                    raise ValueError(
                        "Source file must be inside the corpus directory."
                    )

                relative_path = file_path.relative_to(
                    self.project_root
                ).as_posix()

                file_paths.append(relative_path)

        return sorted(file_paths)

    def load_corpus(
        self,
    ) -> List[SourceFile]:

        source_files = self.discover_source_files()

        if not source_files:
            raise ValueError(
                "No supported source files were found in the corpus directory."
            )

        source_file_records = []

        for source_file in tqdm(
            source_files, desc="Loading corpus", unit="file"
        ):
            source_file = self.read_source_file(source_file)
            source_file_records.append(source_file)

        return source_file_records

    def snapshot_files(self) -> List[CorpusFileState]:
        file_paths = self.discover_source_files()
        snapshots = []

        for file_path in file_paths:
            actual_path = self.project_root / file_path
            metadata = actual_path.stat()

            snapshot = CorpusFileState(
                file_path=file_path,
                size_bytes=metadata.st_size,
                mtime_ns=metadata.st_mtime_ns,
            )
            snapshots.append(snapshot)

        return snapshots
