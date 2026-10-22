import ast
from typing import List
from src.models import SourceFile, IndexSettings, Passage


class Chunker:
    def __init__(self, settings: IndexSettings) -> None:
        self.settings = settings

    def split_span(
        self, source: SourceFile, start: int, end: int
    ) -> List[Passage]:

        passages = []
        current_position = start

        while current_position < end:
            if not 0 <= start <= end <= len(source.text):
                raise ValueError(
                    "Source range must satisfy 0 <= start <= end <= len(source.text)."
                )

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
