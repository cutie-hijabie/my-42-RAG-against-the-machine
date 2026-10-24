from math import log
from typing import Dict, List
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

    def score_candidates(self, query_terms: List[str]) -> Dict[int, float]:
        scores = {}
        n = len(self.index.passages)
        avgdl = self.index.average_document_length
        k1 = self.index.settings.k1
        b = self.index.settings.b

        for term in dict.fromkeys(query_terms):
            postings = self.index.postings.get(term, [])
            if not postings:
                continue
            df = len(postings)
            idf = log(1 + (n - df + 0.5) / (df + 0.5))

            for post in postings:
                chunk_id = post.chunk_id
                tf = post.term_frequency
                dl = self.index.document_lengths[chunk_id]

                contribution = (
                    idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
                )

                scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution

        return scores
