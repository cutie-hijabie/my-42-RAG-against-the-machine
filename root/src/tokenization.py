import re
from typing import List


def tokenize(text: str) -> List[str]:
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
