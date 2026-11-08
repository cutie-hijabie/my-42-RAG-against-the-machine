import os


def validate_chunk_size(max_chunk_size: int) -> int:
    """Check that a passage size is within the supported range.

    Args:
        max_chunk_size: Maximum passage length in characters.

    Returns:
        The original integer if it is between 1 and 2000 inclusive.

    Raises:
        ValueError: If the value is outside the allowed range,
            a boolean, or another non-integer value.
    """
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
    """Check that the requested retrieval limit is valid.

    Args:
        k: Requested maximum number of sources to retrieve.

    Returns:
        The original nonnegative integer. Zero is valid.

    Raises:
        ValueError: If the value is negative, a boolean, or
            another non-integer value.
    """
    if not isinstance(k, int) or (isinstance(k, bool)) or k < 0:
        raise ValueError(
            "k must be a nonnegative integer; booleans are not allowed."
        )

    return k


def validate_query(query: str) -> str:
    """Check that a query contains at least one non-whitespace character.

    Args:
        query: Query text to validate.

    Returns:
        The original query string, including its original whitespace.

    Raises:
        ValueError: If the query is not a string or is empty after
            stripping whitespace.
    """
    if not isinstance(query, str):
        raise ValueError("Query must be a string.")
    if not query.strip():
        raise ValueError("Query must not be empty or whitespace-only.")

    return query


def require_file(file_path: str) -> str:
    """Check that a path points to an existing regular file.

    Args:
        file_path: File path to validate.

    Returns:
        The original path string.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path exists but is not a regular file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file does not exist: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"Input path is not a file: {file_path}")

    return file_path


def require_directory(directory_path: str) -> str:
    """Check that a path points to an existing directory.

    Args:
        directory_path: Directory path to validate.

    Returns:
        The original path string.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path exists but is not a directory.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(
            f"Input directory does not exist: {directory_path}"
        )

    if not os.path.isdir(directory_path):
        raise ValueError(f"Input path is not a directory: {directory_path}")

    return directory_path
