from typing import Protocol


class TextPreprocessor(Protocol):
    def prepare(self, text: str) -> str: ...
