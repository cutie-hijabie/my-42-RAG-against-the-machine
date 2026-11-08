from math import log
from typing import Dict, List
from src.tokenization import tokenize
from pathlib import PurePosixPath
from src.models import IndexData, MinimalSource
from src.validation import validate_k, validate_query


class Retriever:
    """Rank indexed passages and return their source locations.

    Use BM25 to score passages containing query terms. For Markdown
    candidates, add a bonus when query tokens also appear in the file
    path. Return the highest-ranked distinct source locations.

    Attributes:
        index: Loaded index containing the passages, token postings,
            passage lengths, and BM25 settings.
    """

    def __init__(self, index: IndexData) -> None:
        """Initialize the retriever with a loaded index.

        Args:
            index: Validated index to reuse across searches.
        """
        self.index = index

    def search(self, query: str, k: int = 5) -> List[MinimalSource]:
        """Find up to ``k`` ranked source locations for a query.

        Validate the query and retrieval limit, tokenize the query,
        and score matching passages with BM25. For each Markdown
        candidate, add four points per distinct query token shared
        with the file path. Tokenize the path without its extension
        and remove the corpus prefix when present. Reuse the bonus
        for candidates from the same file.

        Keep candidates with positive scores and sort by descending
        score, breaking ties by ascending chunk ID. Treat identical
        file paths and character ranges as one source location.

        Args:
            query: Nonempty search query containing at least one
                non-whitespace character.
            k: Maximum number of distinct source locations to return.
                Defaults to 5. A value of zero returns an empty list
                after input validation.

        Returns:
            Ranked source locations containing file paths and character
            offsets. The list is empty if no candidates match.

        Raises:
            ValueError: If the query is invalid or ``k`` is negative,
                a boolean, or another non-integer value.
        """
        query = validate_query(query)
        k = validate_k(k)

        if k == 0:
            return []

        tokens = tokenize(query)
        scored_candidates = self.score_candidates(tokens)
        query_terms = set(tokens)
        path_bonuses: Dict[str, int] = {}
        corpus_path = PurePosixPath(self.index.corpus_path)

        for chunk_id, score in scored_candidates.items():
            file_path = self.index.passages[chunk_id].file_path
            if not file_path.endswith(".md"):
                continue

            if file_path not in path_bonuses:
                path = PurePosixPath(file_path)
                if path.is_relative_to(corpus_path):
                    path = path.relative_to(corpus_path)

                path_terms = set(tokenize(path.with_suffix("").as_posix()))
                path_bonuses[file_path] = 4 * len(query_terms & path_terms)

            scored_candidates[chunk_id] = score + path_bonuses[file_path]
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
        """Calculate base BM25 scores for matching passages.

        Look up each distinct query token in the inverted index.
        Calculate its inverse document frequency from the number of
        passages containing it, then score each matching passage using
        the token's frequency and the passage's length relative to
        the average. Sum the contributions for each chunk ID.

        Args:
            query_terms: Tokens produced by the query tokenizer.
                Repeated query tokens contribute once each.

        Returns:
            A mapping from matching chunk IDs to their BM25 scores.
            The mapping is empty if none of the query terms appear
            in the index.
        """
        scores: Dict = {}
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
