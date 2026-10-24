import os


def validate_chunk_size(max_chunk_size: int) -> int:
    if (
        not isinstance(max_chunk_size, int)
        or (max_chunk_size < 1 or max_chunk_size > 2000)
        or (isinstance(max_chunk_size, bool))
    ):
        raise ValueError(
            "max_chunk_size must be an integer between 1 and 2000;"
            " booleans are not allowed."
        )

    return max_chunk_size


def validate_k(k: int) -> int:
    if not isinstance(k, int) or (isinstance(k, bool)) or k < 0:
        raise ValueError(
            "k must be a nonnegative integer; booleans are not allowed."
        )

    return k


def validate_query(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("Query must be a string.")
    if not query.strip():
        raise ValueError("Query must not be empty or whitespace-only.")

    return query


def require_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file does not exist: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"Input path is not a file: {file_path}")

    return file_path


def require_directory(directory_path: str) -> str:
    if not os.path.exists(directory_path):
        raise FileNotFoundError(
            f"Input directory does not exist: {directory_path}"
        )

    if not os.path.isdir(directory_path):
        raise ValueError(f"Input path is not a directory: {directory_path}")

    return directory_path
