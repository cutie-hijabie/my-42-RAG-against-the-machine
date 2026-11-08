import os
import ast
from typing import List
from src.models import SourceFile, IndexSettings, Passage


class Chunker:
    """Split source files into passages for indexing and retrieval.

    Use top-level definitions as boundaries in Python files and
    headings or blank lines as boundaries in text documents. Split
    oversized sections into smaller passages using the configured
    character limit and overlap.

    Each passage keeps its source path and half-open character range
    so that slicing the original text at that range reproduces the
    passage.

    Attributes:
        settings: Index settings that provide the maximum passage
            size and requested overlap, both measured in characters.
    """

    def __init__(self, settings: IndexSettings) -> None:
        """Initialize the chunker with the supplied index settings.

        Args:
            settings: Validated index settings. The chunker uses
                ``max_chunk_size`` and ``overlap`` when splitting
                source text.
        """
        self.settings = settings

    def split_span(
        self, source: SourceFile, start: int, end: int
    ) -> List[Passage]:
        """Split a character range into size-limited passages.

        Start each passage at the current position and extend it up
        to the configured character limit or the end of the range.
        If more text remains, prefer ending just after the last
        newline within that window when one is available.

        Start the next passage with up to the requested overlap.
        Advance by at least one character on each iteration so that
        splitting makes progress even when a newline produces a
        passage shorter than the requested overlap.

        Args:
            source: Source file containing the text and its path.
            start: Inclusive character offset of the range to split.
            end: Exclusive character offset of the range to split.

        Returns:
            Passages in source order, each with its original path,
            character offsets, and text. An empty range produces
            an empty list.

        Raises:
            ValueError: If the offsets do not satisfy
                ``0 <= start <= end <= len(source.text)``.
        """

        if not 0 <= start <= end <= len(source.text):
            raise ValueError(
                "Source range must satisfy 0 <= start <= end "
                "<= len(source.text)."
            )

        passages = []
        current_position = start

        while current_position < end:
            stop = min(self.settings.max_chunk_size + current_position, end)

            if stop < end:
                newline_postion = source.text.rfind(
                    "\n", current_position, stop
                )
                if newline_postion != -1:
                    stop = newline_postion + 1

            passage = Passage(
                file_path=source.file_path,
                first_character_index=current_position,
                last_character_index=stop,
                text=source.text[current_position:stop],
            )
            passages.append(passage)

            if stop == end:
                break
            else:
                current_position = max(
                    current_position + 1, stop - self.settings.overlap
                )

        return passages

    def chunk_python(self, source: SourceFile) -> List[Passage]:
        """Split a Python source file at top-level definitions.

        Parse the source with ``ast`` to locate top-level classes,
        functions, and async functions. Use the earliest decorator
        line as the boundary for decorated definitions. Convert line
        numbers to character offsets in the original source text.

        Keep each section together when it fits the character limit.
        Delegate oversized sections to ``split_span``. If parsing
        raises ``SyntaxError``, split the entire file using that same
        method.

        Args:
            source: Python source file to split.

        Returns:
            Passages in source order. An empty source produces an
            empty list.
        """
        if not source.text:
            return []

        try:
            parsed_files = ast.parse(source.text)
        except SyntaxError:
            return self.split_span(source, 0, len(source.text))

        start_postitions = []
        pos = 0
        lines = source.text.splitlines(keepends=True)

        for line in lines:
            start_postitions.append(pos)
            pos += len(line)

        boundaries = {0, len(source.text)}

        for file in parsed_files.body:
            if isinstance(
                file, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                start_line = file.lineno
                if file.decorator_list:
                    start_line = min(
                        decorator.lineno for decorator in file.decorator_list
                    )

                boundaries.add(start_postitions[start_line - 1])

        sorted_boundaries = sorted(boundaries)

        passages = []

        for i, boundary in enumerate(sorted_boundaries[:-1]):
            start = boundary
            end = sorted_boundaries[i + 1]
            section_length = end - start

            if section_length <= self.settings.max_chunk_size:
                passage = Passage(
                    file_path=source.file_path,
                    first_character_index=start,
                    last_character_index=end,
                    text=source.text[start:end],
                )
                passages.append(passage)
            else:
                passages.extend(self.split_span(source, start, end))

        return passages

    def chunk_text(self, source: SourceFile) -> List[Passage]:
        """Split a text document at headings and paragraph breaks.

        Recognize hash-style headings, use blank lines as additional
        boundaries for files other than Markdown, and recognize
        reStructuredText headings underlined with equals signs or
        hyphens.

        Track backtick and tilde fences during the boundary scan so
        that lines inside fenced blocks do not create section
        boundaries. After scanning, split any oversized section with
        ``split_span`` to enforce the character limit.

        Args:
            source: Markdown, plain-text, or reStructuredText file
                to split.

        Returns:
            Passages in source order. An empty source produces an
            empty list.
        """
        if not source.text:
            return []

        boundaries = {0, len(source.text)}

        fence_character = None
        fence_length = 0
        start_postitions = []
        pos = 0
        previous_line = ""
        previous_position = 0
        lines = source.text.splitlines(keepends=True)

        for line in lines:
            start_postitions.append(pos)
            stripped = line.lstrip()

            if fence_character is None:
                if stripped.startswith(("```", "~~~")):
                    fence_character = stripped[0]
                    fence_length = len(stripped) - len(
                        stripped.lstrip(fence_character)
                    )
                else:
                    heading_length = len(stripped) - len(stripped.lstrip("#"))

                    if 1 <= heading_length <= 6 and (
                        heading_length == len(stripped)
                        or stripped[heading_length].isspace()
                    ):
                        boundaries.add(pos)

                    if not line.strip() and not source.file_path.endswith(
                        ".md"
                    ):
                        boundaries.add(pos + len(line))

                    underline = line.strip()
                    if (
                        source.file_path.endswith(".rst")
                        and previous_line.strip()
                        and len(underline) >= 3
                        and underline[0] in "=-"
                        and all(
                            character == underline[0]
                            for character in underline
                        )
                    ):
                        boundaries.add(previous_position)

            else:
                closing_length = len(stripped) - len(
                    stripped.lstrip(fence_character)
                )
                if (
                    closing_length >= fence_length
                    and not stripped[closing_length:].strip()
                ):
                    fence_character = None
                    fence_length = 0

            previous_line = line
            previous_position = pos

            pos += len(line)

        sorted_boundaries = sorted(boundaries)

        passages = []

        for i, boundary in enumerate(sorted_boundaries[:-1]):
            start = boundary
            end = sorted_boundaries[i + 1]
            section_length = end - start

            if section_length <= self.settings.max_chunk_size:
                passage = Passage(
                    file_path=source.file_path,
                    first_character_index=start,
                    last_character_index=end,
                    text=source.text[start:end],
                )
                passages.append(passage)
            else:
                passages.extend(self.split_span(source, start, end))

        return passages

    def chunk_source(self, source: SourceFile) -> List[Passage]:
        """Choose a chunking strategy based on the file extension.

        Send ``.py`` files to the Python chunker and ``.md``,
        ``.txt``, and ``.rst`` files to the text chunker.

        Args:
            source: Source file containing the path and text to split.

        Returns:
            The passages produced by the selected chunking strategy.

        Raises:
            ValueError: If the file extension is unsupported.
        """
        _, extension = os.path.splitext(source.file_path)
        if extension == ".py":
            return self.chunk_python(source)
        elif extension in (".md", ".txt", ".rst"):
            return self.chunk_text(source)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
