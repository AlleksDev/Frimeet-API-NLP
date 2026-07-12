from app.shared.nlp.preprocessing.text import prepare_for_embedding


class SharedTextPreprocessor:
    def prepare(self, text: str) -> str:
        return prepare_for_embedding(text)
