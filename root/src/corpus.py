import os
from tqdm import tqdm
from typing import List
from pathlib import Path
from src.models import SourceFile, CorpusFileState
from src.validation import require_directory, require_file


def raise_walk_error(error: OSError) -> None:
    """Propagate a filesystem error reported by ``os.walk``.

    Use this function as the walk's ``onerror`` callback so that
    directory traversal stops and the caller receives the original
    error.

    Args:
        error: The filesystem error reported by ``os.walk``.

    Raises:
        OSError: The same exception supplied in ``error``.
    """
    raise error


class CorpusLoader:
    """Discover, read, and capture metadata for corpus files.

    Use the project root as the base for relative paths. Represent
    source paths relative to that root with forward slashes so that
    indexing and passage reconstruction use the same path format.

    Attributes:
        project_root: Absolute, resolved path to the project root.
        corpus_path: Corpus directory as a project-relative POSIX path.
    """

    def __init__(self, corpus_path: str = "data/raw/vllm-0.10.1") -> None:
        """Initialize the loader for an existing corpus directory.

        Derive the project root from this module's location and
        interpret relative corpus paths from that root. Check the
        corpus path and its parents for symbolic links before
        resolving it. Verify that the resolved path is an existing
        directory inside the project root, then store it as a
        project-relative POSIX path.

        Args:
            corpus_path: Absolute or project-relative path to the
                corpus directory. Defaults to the supplied vLLM corpus.

        Raises:
            FileNotFoundError: If the corpus directory does not exist.
            ValueError: If the path contains a symbolic link, resolves
                outside the project root, or points to a non-directory.
        """
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
        """Read one source file and return its path and text.

        Resolve the path against the project root when it is relative,
        then verify that it points to a regular file inside the
        project. Read the file as UTF-8 while preserving its original
        newline sequences so that stored character offsets can be
        reused when reconstructing passages.

        Args:
            file_path: Absolute or project-relative path to the file.

        Returns:
            A source record containing the project-relative POSIX
            path and the complete decoded text.

        Raises:
            FileNotFoundError: If the source file does not exist.
            ValueError: If the resolved path is outside the project
                root or points to something other than a regular file.
            UnicodeDecodeError: If the file is not valid UTF-8.
            OSError: If the file cannot be opened or read.
        """

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
        """Find supported source files under the corpus directory.

        Walk the corpus recursively and skip hidden directories and
        symbolic links. Include regular files ending in ``.py``,
        ``.md``, ``.txt``, or ``.rst``. Resolve each selected path
        and verify that it remains inside the corpus directory.

        Returns:
            A sorted list of project-relative POSIX file paths. The
            list is empty when no supported files are found.

        Raises:
            OSError: If directory traversal or a filesystem check
                fails.
            ValueError: If a selected file resolves outside the
                corpus directory.
        """

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
        """Load all supported corpus files into source records.

        Discover the source paths, then read each file in sorted
        order while displaying a progress bar. Preserve the order
        of the discovered paths in the returned records.

        Returns:
            A list of source records containing the relative path
            and full text of each file.

        Raises:
            ValueError: If no supported files are found or a source
                path fails validation.
            UnicodeDecodeError: If a source file is not valid UTF-8.
            OSError: If discovery fails or a source file cannot be
                opened or read.
        """
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
        """Capture filesystem metadata for the current corpus files.

        Discover the supported files and collect each file's
        project-relative path, size in bytes, and modification time
        in nanoseconds. Callers can compare snapshots to detect
        added or removed files and changes to the recorded metadata.

        Returns:
            A list of file-state records in sorted path order.

        Raises:
            OSError: If directory traversal fails or file metadata
                cannot be read.
            ValueError: If a discovered path resolves outside the
                corpus directory.
        """
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
