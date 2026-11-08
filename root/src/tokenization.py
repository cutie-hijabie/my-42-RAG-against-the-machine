import re
from typing import List


def tokenize(text: str) -> List[str]:
    """Convert text into lowercase tokens for lexical retrieval.

    Keep each word or identifier as a lowercase token. Also emit
    distinct parts from underscore-separated and mixed-case
    identifiers so a query can match either the full identifier
    or one of its parts.

    Deduplicate parts within each identifier occurrence. Preserve
    repeated occurrences across the input so callers can calculate
    term frequencies from the returned list.

    Args:
        text: Source text, search query, or file path to tokenize.

    Returns:
        Lowercase tokens in encounter order, with each identifier
        followed by its additional parts.
    """
    tokens = []
    identifiers = re.findall(r"\w+", text)

    for identifier in identifiers:
        tokens.append(identifier.lower())

        separated = re.sub(r"([A-Z])([A-Z][a-z])", r"\1_\2", identifier)
        separated = re.sub(r"([a-z])([A-Z])", r"\1_\2", separated)
        parts = separated.split("_")

        seen_parts = set()

        for part in parts:
            if not part:
                continue
            lower_part = part.lower()
            if (
                lower_part != identifier.lower()
                and lower_part not in seen_parts
            ):
                tokens.append(lower_part)
                seen_parts.add(lower_part)

    return tokens
