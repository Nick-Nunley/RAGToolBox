import re
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Protocol

class Chunker(Protocol):

    def chunk(self, text: str) -> List[str]:
        ...

    @staticmethod
    def _find_next_word_boundary(text: str, idx: int) -> int:
        text_len = len(text)
        # Move forward to next whitespace
        while idx < text_len and not text[idx].isspace():
            idx += 1
        # Move past whitespace to start of next word
        while idx < text_len and text[idx].isspace():
            idx += 1
        return idx

    @staticmethod
    def _find_prev_word_boundary(text: str, idx: int, min_idx: int = 0) -> int:
        # Move backward to previous whitespace
        while idx > min_idx and not text[idx].isspace():
            idx -= 1
        return idx

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
        Split text into overlapping chunks using sliding window, ensuring chunks start and end at word boundaries.
        Args:
            text: Input text to chunk
        Returns:
            List of overlapping text chunks
        """
        if not text.strip():
            return []

        chunks = []
        start = 0
        text_len = len(text)
        prev_end = 0

        while start < text_len:
            end = start + self.window_size
            if end > text_len:
                end = text_len

            # Adjust start and end using Chunker static helpers
            min_start = prev_end if chunks else 0
            adj_start = Chunker._find_next_word_boundary(text, max(start, min_start))
            adj_end = Chunker._find_next_word_boundary(text, end)

            # Extract the chunk and clean it up
            chunk = text[adj_start:adj_end].strip()
            if chunk:
                if not chunks or chunk != chunks[-1]:
                    chunks.append(chunk)

            prev_end = adj_end

            # Move to next chunk position with overlap
            start = end - self.overlap
            if start < prev_end:
                start = prev_end

        return chunks

class SectionAwareChunker(Chunker):
    """
    Chunks text while preserving section structure and metadata.
    Useful for RAG systems that need to maintain document structure.
    """
    
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the section-aware chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks
        """
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= max_chunk_size:
            raise ValueError("overlap must be less than max_chunk_size")
        
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks while preserving section structure.
        
        Args:
            text: Input text with markdown headers
            
        Returns:
            List of text chunks with section context
        """
        if not text.strip():
            return []
        
        # Split by markdown headers
        sections = re.split(r'(^#{1,6}\s+.+$)', text, flags=re.MULTILINE)
        
        chunks = []
        current_header = ""
        i = 0
        while i < len(sections):
            section = sections[i]
            if not section.strip():
                i += 1
                continue
                
            # Check if this is a header
            if re.match(r'^#{1,6}\s+', section.strip()):
                current_header = section.strip()
                # Check if next section is content or another header/end
                if i+1 >= len(sections) or re.match(r'^#{1,6}\s+', sections[i+1].strip()) or not sections[i+1].strip():
                    # No content for this header, emit header-only chunk
                    chunks.append(current_header)
                i += 1
                continue
            
            # This is content, combine with header
            full_section = f"{current_header}\n{section}" if current_header else section
            
            # If section is too large, split it further
            if len(full_section) > self.max_chunk_size:
                sub_chunks = self._split_large_section(full_section, current_header)
                chunks.extend(sub_chunks)
            else:
                chunks.append(full_section.strip())
            i += 1
        
        return [chunk for chunk in chunks if chunk.strip()]

    def _split_large_section(self, section_text: str, header: str) -> List[str]:
        """
        Splits a large section into smaller chunks while preserving context.
        """
        chunks = []
        remaining = section_text
        original_length = len(section_text)
        
        while len(remaining) > self.max_chunk_size:
            # Find a good break point
            break_point = self._find_break_point(remaining, self.max_chunk_size)
            # Use superclass static method to avoid splitting words
            break_point = Chunker._find_next_word_boundary(remaining, break_point)
            
            # Extract chunk and add header context
            chunk = remaining[:break_point].strip()
            if header and not chunk.startswith(header):
                chunk = f"{header}\n{chunk}"
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            overlap_start = max(0, break_point - self.overlap)
            # Adjust overlap_start to next word boundary
            overlap_start = Chunker._find_next_word_boundary(remaining, overlap_start)
            remaining = remaining[overlap_start:]
            
            # Safety check: if we're not making progress, force advancement
            if len(remaining) >= original_length or overlap_start == 0:
                # Force advancement by at least one character
                remaining = remaining[1:] if len(remaining) > 1 else ""
            
            # If we're not making progress, force advancement
            if len(remaining) >= original_length:
                break
        
        # Add remaining text
        if remaining.strip():
            chunk = remaining.strip()
            if header and not chunk.startswith(header):
                chunk = f"{header}\n{chunk}"
            chunks.append(chunk)
        
        return chunks

    def _find_break_point(self, text: str, max_size: int) -> int:
        """
        Finds a good break point within the text.
        """
        # Look for paragraph breaks first
        search_start = max(0, max_size - 200)
        last_paragraph = text.rfind('\n\n', search_start, max_size)
        
        if last_paragraph > search_start:
            return last_paragraph + 2        
        # Look for sentence breaks
        last_sentence = text.rfind('.', search_start, max_size)
        if last_sentence > search_start:
            return last_sentence + 2        
        # Look for word breaks
        last_space = text.rfind(' ', search_start, max_size)
        if last_space > search_start:
            return last_space + 1        
        # If all else fails, just break at max_size
        return max_size

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
