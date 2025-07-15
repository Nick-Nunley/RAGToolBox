import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Protocol

class Chunker(Protocol):

    def chunk(self, text: str) -> List[str]:
        ...

class ParagraphChunker(Chunker):

    def chunk(self, text: str) -> List[str]:
        # Implementation: split by double newlines
        return [p.strip() for p in text.split('\n\n') if p.strip()]

class SentenceChunker(Chunker):

    def chunk(self, text: str) -> List[str]:
        # Check if punkt data is already downloaded, download if not
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        return sent_tokenize(text)
