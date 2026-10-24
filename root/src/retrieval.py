from typing import List
from src.tokenization import tokenize
from src.models import IndexData, MinimalSource
from src.validation import validate_k, validate_query


class Retriever:
    def __init__(self, index: IndexData) -> None:
        self.index = index

    def search(self, query: str, k: int = 5) -> List[MinimalSource]:
        query = validate_query(query)
        k = validate_k(k)

        if k == 0:
            return []

        tokens = tokenize(query)
        scored_candidates = self.score_candidates(tokens)
        filtered_candidates = {
            chunk_id: score
            for chunk_id, score in scored_candidates.items()
            if score > 0
        }

        sorted_candidates = sorted(
            filtered_candidates.items(),
            key=lambda item: (-item[1], item[0]),
        )

        final_candidates = []
        source_locations = set()

        for chunk_id, score in sorted_candidates:
            passage = self.index.passages[chunk_id]
            location = (
                passage.file_path,
                passage.first_character_index,
                passage.last_character_index,
            )

            if location not in source_locations:
                source_locations.add(location)

                source = MinimalSource(
                    file_path=passage.file_path,
                    first_character_index=passage.first_character_index,
                    last_character_index=passage.last_character_index,
                )

                final_candidates.append(source)

            if len(final_candidates) == k:
                break

        return final_candidates

    def score_candidates
