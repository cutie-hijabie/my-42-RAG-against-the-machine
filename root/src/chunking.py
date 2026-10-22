import os
import ast
from typing import List
from src.models import SourceFile, IndexSettings, Passage


class Chunker:
    def __init__(self, settings: IndexSettings) -> None:
        self.settings = settings

    def split_span(
        self, source: SourceFile, start: int, end: int
    ) -> List[Passage]:

        if not 0 <= start <= end <= len(source.text):
            raise ValueError(
                "Source range must satisfy 0 <= start <= end <= len(source.text)."
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

                    if not line.strip():
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
        _, extension = os.path.splitext(source.file_path)
        if extension == ".py":
            return self.chunk_python(source)
        elif extension in (".md", ".txt", ".rst"):
            return self.chunk_text(source)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
