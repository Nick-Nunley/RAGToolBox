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

class SlidingWindowChunker(Chunker):
    """
    Creates overlapping chunks using a sliding window approach.
    Useful for RAG systems to maintain context across chunk boundaries.
    """
    
    def __init__(self, window_size: int = 1000, overlap: int = 200):
        """
        Initialize the sliding window chunker.
        
        Args:
            window_size: Maximum number of characters per chunk
            overlap: Number of characters to overlap between consecutive chunks
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= window_size:
            raise ValueError("overlap must be less than window_size")
        
        self.window_size = window_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks using sliding window.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of overlapping text chunks
        """
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position for current chunk
            end = start + self.window_size
            
            # If this is not the last chunk, try to break at a word boundary
            if end < len(text):
                # Look for the last space or newline within the last 100 characters
                # to avoid breaking words in the middle
                search_start = max(start + self.window_size - 100, start)
                last_space = text.rfind(' ', search_start, end)
                last_newline = text.rfind('\n', search_start, end)
                
                # Use the later of space or newline, or just use end if neither found
                break_point = max(last_space, last_newline)
                if break_point > start:
                    end = break_point + 1  # Include the space/newline
            
            # Extract the chunk and clean it up
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move to next chunk position with overlap
            start = end - self.overlap
            
            # If we're not making progress, force advancement
            if start >= end:
                start = end
        
        return chunks

class HierarchicalChunker(Chunker):
    """
    Applies a sequence of chunkers hierarchically. The output of one chunker is fed as input to the next.
    Useful for multi-stage chunking (e.g., paragraph -> sliding window).
    """
    def __init__(self, chunkers: List[Chunker]):
        if not chunkers:
            raise ValueError("At least one chunker must be provided")
        self.chunkers = chunkers

    def chunk(self, text: str) -> List[str]:
        chunks = [text]
        for chunker in self.chunkers:
            new_chunks = []
            for chunk in chunks:
                new_chunks.extend(chunker.chunk(chunk))
            chunks = new_chunks
        return chunks
