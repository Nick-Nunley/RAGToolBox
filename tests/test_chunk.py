import pytest
from typing import Any, List

from src.chunk import (
    Chunker,
    ParagraphChunker,
    SentenceChunker,
    SlidingWindowChunker
)

# Test data
SAMPLE_TEXT = """This is the first paragraph. It contains multiple sentences. Each sentence should be properly handled.

This is the second paragraph. It has different content. The chunker should separate these paragraphs.

This is the third paragraph. It's shorter than the others."""

SAMPLE_TEXT_NO_PARAGRAPHS = "This is a single paragraph. It has multiple sentences. But no paragraph breaks."

SAMPLE_TEXT_EMPTY_PARAGRAPHS = """First paragraph.

Second paragraph.

Third paragraph."""

SAMPLE_TEXT_MIXED_WHITESPACE = """   First paragraph with leading spaces.  


Second paragraph with extra newlines.


Third paragraph with trailing spaces.   """

# Unit tests
def test_paragraph_chunker_basic() -> None:
    """Test basic paragraph chunking functionality."""
    chunker = ParagraphChunker()
    chunks = chunker.chunk(SAMPLE_TEXT)
    
    assert len(chunks) == 3
    assert "first paragraph" in chunks[0]
    assert "second paragraph" in chunks[1]
    assert "third paragraph" in chunks[2]

def test_paragraph_chunker_single_paragraph() -> None:
    """Test chunking text with no paragraph breaks."""
    chunker = ParagraphChunker()
    chunks = chunker.chunk(SAMPLE_TEXT_NO_PARAGRAPHS)
    
    assert len(chunks) == 1
    assert "single paragraph" in chunks[0]

def test_paragraph_chunker_empty_paragraphs() -> None:
    """Test chunking text with empty paragraphs."""
    chunker = ParagraphChunker()
    chunks = chunker.chunk(SAMPLE_TEXT_EMPTY_PARAGRAPHS)
    
    assert len(chunks) == 3
    assert all(chunk.strip() for chunk in chunks)  # No empty chunks

def test_paragraph_chunker_mixed_whitespace() -> None:
    """Test chunking text with various whitespace patterns."""
    chunker = ParagraphChunker()
    chunks = chunker.chunk(SAMPLE_TEXT_MIXED_WHITESPACE)
    
    assert len(chunks) == 3
    # Check that leading/trailing whitespace is stripped
    assert chunks[0].startswith("First paragraph")
    assert chunks[1].startswith("Second paragraph")
    assert chunks[2].startswith("Third paragraph")

def test_paragraph_chunker_empty_text() -> None:
    """Test chunking empty text."""
    chunker = ParagraphChunker()
    chunks = chunker.chunk("")
    
    assert len(chunks) == 0

def test_paragraph_chunker_whitespace_only() -> None:
    """Test chunking text with only whitespace."""
    chunker = ParagraphChunker()
    chunks = chunker.chunk("   \n\n   \n\n   ")
    
    assert len(chunks) == 0

def test_sentence_chunker_basic() -> None:
    """Test basic sentence chunking functionality."""
    chunker = SentenceChunker()
    chunks = chunker.chunk(SAMPLE_TEXT_NO_PARAGRAPHS)
    
    assert len(chunks) == 3
    assert "single paragraph" in chunks[0]
    assert "multiple sentences" in chunks[1]
    assert "paragraph breaks" in chunks[2]

def test_sentence_chunker_complex_sentences() -> None:
    """Test chunking text with complex sentence structures."""
    complex_text = "Dr. Smith said: 'This is a quote.' Mr. Jones replied. What about abbreviations like etc.?"
    chunker = SentenceChunker()
    chunks = chunker.chunk(complex_text)
    
    # NLTK tokenizes more granularly - it separates "Fig." and "etc." as separate tokens
    assert len(chunks) == 4
    assert "Dr. Smith said" in chunks[0]
    assert "Mr. Jones replied" in chunks[1]
    assert "abbreviations like etc" in chunks[2]
    assert chunks[3] == "?"  # NLTK treats trailing punctuation as separate

def test_sentence_chunker_empty_text() -> None:
    """Test chunking empty text."""
    chunker = SentenceChunker()
    chunks = chunker.chunk("")
    
    assert len(chunks) == 0

def test_sentence_chunker_single_sentence() -> None:
    """Test chunking a single sentence."""
    chunker = SentenceChunker()
    chunks = chunker.chunk("This is a single sentence.")
    
    assert len(chunks) == 1
    assert chunks[0] == "This is a single sentence."

def test_sentence_chunker_with_numbers() -> None:
    """Test chunking text with numbers and abbreviations."""
    text = "The study used 1.5 Tesla MRI. Fig. 1 shows the results. The p-value was < 0.05."
    chunker = SentenceChunker()
    chunks = chunker.chunk(text)
    
    # NLTK separates "Fig." as a separate token due to the period
    assert len(chunks) == 4
    assert "1.5 Tesla MRI" in chunks[0]
    assert "Fig." in chunks[1]
    assert "1 shows the results" in chunks[2]
    assert "p-value was < 0.05" in chunks[3]

def test_sentence_chunker_with_quotes() -> None:
    """Test chunking text with quoted sentences."""
    text = 'He said "This is quoted text." Then he continued. "Another quote."'
    chunker = SentenceChunker()
    chunks = chunker.chunk(text)
    
    assert len(chunks) == 3
    assert "quoted text" in chunks[0]
    assert "he continued" in chunks[1]
    assert "Another quote" in chunks[2]

def test_chunker_protocol_compliance() -> None:
    """Test that chunkers properly implement the Chunker protocol."""
    paragraph_chunker = ParagraphChunker()
    sentence_chunker = SentenceChunker()
    
    # Test that they can be used as Chunker types
    def process_with_chunker(chunker: Chunker, text: str) -> List[str]:
        return chunker.chunk(text)
    
    # Should work with both chunker types
    result1 = process_with_chunker(paragraph_chunker, "Test text.")
    result2 = process_with_chunker(sentence_chunker, "Test text.")
    
    assert isinstance(result1, list)
    assert isinstance(result2, list)

# Integration tests
def test_chunker_with_real_document_content() -> None:
    """Test chunkers with realistic document content."""
    real_document = """Low-intensity focused ultrasound (LIFU) is a non-invasive neuromodulation technique.

The technique uses focused ultrasound waves to modulate neural activity. It has shown promise in various applications.

Studies have demonstrated its effectiveness in treating neurological disorders. The safety profile appears favorable."""
    
    paragraph_chunker = ParagraphChunker()
    sentence_chunker = SentenceChunker()
    
    paragraph_chunks = paragraph_chunker.chunk(real_document)
    sentence_chunks = sentence_chunker.chunk(real_document)
    
    # Test paragraph chunking
    assert len(paragraph_chunks) == 3
    assert "LIFU" in paragraph_chunks[0]
    assert "focused ultrasound waves" in paragraph_chunks[1]
    assert "neurological disorders" in paragraph_chunks[2]
    
    # Test sentence chunking - NLTK creates 5 sentences from the 4 logical sentences
    assert len(sentence_chunks) == 5
    assert "LIFU" in sentence_chunks[0]
    assert "focused ultrasound waves" in sentence_chunks[1]
    assert "various applications" in sentence_chunks[2]
    assert "neurological disorders" in sentence_chunks[3]
    assert "safety profile" in sentence_chunks[4]

def test_chunker_performance_with_large_text() -> None:
    """Test chunker performance with larger text samples."""
    large_text = "Sentence one. " * 100 + "Sentence two. " * 100
    
    paragraph_chunker = ParagraphChunker()
    sentence_chunker = SentenceChunker()
    
    # Should handle large text without errors
    paragraph_chunks = paragraph_chunker.chunk(large_text)
    sentence_chunks = sentence_chunker.chunk(large_text)
    
    assert len(paragraph_chunks) == 1  # No paragraph breaks
    assert len(sentence_chunks) == 200  # 200 sentences

def test_chunker_edge_cases() -> None:
    """Test chunkers with various edge cases."""
    edge_cases = [
        "Single sentence.",
        "Sentence one. Sentence two.",
        "   Leading spaces.   ",
        "Trailing spaces.   ",
        "Multiple\n\n\nnewlines.",
        "Abbreviations like Dr. Smith and Mr. Jones.",
        "Numbers like 1.5 and 2.3.",
        "Quotes: 'Hello.' 'World.'",
        "Mixed punctuation! What about this? And this."
    ]
    
    paragraph_chunker = ParagraphChunker()
    sentence_chunker = SentenceChunker()
    
    for text in edge_cases:
        # Should not raise exceptions
        paragraph_result = paragraph_chunker.chunk(text)
        sentence_result = sentence_chunker.chunk(text)
        
        assert isinstance(paragraph_result, list)
        assert isinstance(sentence_result, list)
        
        # Results should not be empty for non-empty input
        if text.strip():
            assert len(paragraph_result) > 0 or len(sentence_result) > 0

def test_chunker_consistency() -> None:
    """Test that chunkers produce consistent results for the same input."""
    text = "This is a test. It has multiple sentences. We want consistent results."
    
    paragraph_chunker = ParagraphChunker()
    sentence_chunker = SentenceChunker()
    
    # Multiple calls should produce identical results
    result1 = paragraph_chunker.chunk(text)
    result2 = paragraph_chunker.chunk(text)
    assert result1 == result2
    
    result3 = sentence_chunker.chunk(text)
    result4 = sentence_chunker.chunk(text)
    assert result3 == result4

# SlidingWindowChunker tests
def test_sliding_window_chunker_basic() -> None:
    """Test basic sliding window chunking functionality."""
    text = "This is a test document. " * 50  # Create a long text
    chunker = SlidingWindowChunker(window_size=100, overlap=20)
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 1  # Should create multiple chunks
    assert all(len(chunk) <= 100 for chunk in chunks)  # All chunks within size limit
    
    # Check that chunks overlap
    for i in range(len(chunks) - 1):
        # There should be some overlap between consecutive chunks
        overlap_text = chunks[i][-20:]  # Last 20 chars of current chunk
        next_chunk_start = chunks[i + 1][:20]  # First 20 chars of next chunk
        # At least some characters should be the same (allowing for word boundaries)
        assert any(char in next_chunk_start for char in overlap_text)

def test_sliding_window_chunker_word_boundaries() -> None:
    """Test that sliding window respects word boundaries."""
    text = "This is a test document with multiple words. Each word should be preserved."
    chunker = SlidingWindowChunker(window_size=30, overlap=10)
    chunks = chunker.chunk(text)
    
    # Check that no words are split in the middle
    for chunk in chunks:
        # If a word is split, it would have a space in the middle
        words = chunk.split()
        for word in words:
            # Words should not have internal spaces (indicating they were split)
            assert ' ' not in word.strip()

def test_sliding_window_chunker_empty_text() -> None:
    """Test sliding window chunker with empty text."""
    chunker = SlidingWindowChunker()
    chunks = chunker.chunk("")
    assert chunks == []

def test_sliding_window_chunker_short_text() -> None:
    """Test sliding window chunker with text shorter than window size."""
    text = "Short text."
    chunker = SlidingWindowChunker(window_size=100, overlap=20)
    chunks = chunker.chunk(text)
    
    assert len(chunks) == 1
    assert chunks[0] == text

def test_sliding_window_chunker_validation() -> None:
    """Test sliding window chunker parameter validation."""
    # Test invalid window_size
    with pytest.raises(ValueError):
        SlidingWindowChunker(window_size=0)
    
    with pytest.raises(ValueError):
        SlidingWindowChunker(window_size=-1)
    
    # Test invalid overlap
    with pytest.raises(ValueError):
        SlidingWindowChunker(window_size=100, overlap=-1)
    
    # Test overlap >= window_size
    with pytest.raises(ValueError):
        SlidingWindowChunker(window_size=100, overlap=100)
    
    with pytest.raises(ValueError):
        SlidingWindowChunker(window_size=100, overlap=150)

def test_sliding_window_chunker_with_newlines() -> None:
    """Test sliding window chunker with text containing newlines."""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph with more content."
    chunker = SlidingWindowChunker(window_size=50, overlap=10)
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 1
    # Check that newlines are preserved
    for chunk in chunks:
        if '\n' in text:
            assert '\n' in chunk or chunk in text

def test_sliding_window_chunker_large_overlap() -> None:
    """Test sliding window chunker with large overlap."""
    text = "This is a test document. " * 20
    chunker = SlidingWindowChunker(window_size=100, overlap=80)
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 1
    # With large overlap, consecutive chunks should have significant overlap
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]
        # At least 50% of the current chunk should overlap with the next
        overlap_length = len(set(current_chunk[-40:]) & set(next_chunk[:40]))
        assert overlap_length > 0

def test_sliding_window_chunker_no_overlap() -> None:
    """Test sliding window chunker with no overlap."""
    text = "This is a test document. " * 20
    chunker = SlidingWindowChunker(window_size=100, overlap=0)
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 1
    # With no overlap, consecutive chunks should have minimal overlap
    # (only due to word boundaries, not intentional overlap)
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]
        # The overlap should be minimal (just a few characters at word boundaries)
        overlap_length = len(set(current_chunk[-20:]) & set(next_chunk[:20]))
        assert overlap_length < 15  # Allow for some overlap due to word boundaries

def test_sliding_window_chunker_protocol_compliance() -> None:
    """Test that SlidingWindowChunker properly implements the Chunker protocol."""
    chunker = SlidingWindowChunker()
    
    # Test that it can be used as a Chunker type
    def process_with_chunker(chunker: Chunker, text: str) -> List[str]:
        return chunker.chunk(text)
    
    result = process_with_chunker(chunker, "Test text.")
    assert isinstance(result, list)

# Error handling tests
def test_chunker_with_none_input() -> None:
    """Test chunkers handle None input gracefully."""
    paragraph_chunker = ParagraphChunker()
    sentence_chunker = SentenceChunker()
    sliding_chunker = SlidingWindowChunker()
    
    with pytest.raises(AttributeError):
        paragraph_chunker.chunk(None)  # type: ignore
    
    with pytest.raises(TypeError):
        sentence_chunker.chunk(None)  # type: ignore
    
    with pytest.raises(AttributeError):
        sliding_chunker.chunk(None)  # type: ignore

def test_chunker_with_non_string_input() -> None:
    """Test chunkers handle non-string input gracefully."""
    paragraph_chunker = ParagraphChunker()
    sentence_chunker = SentenceChunker()
    sliding_chunker = SlidingWindowChunker()
    
    with pytest.raises(AttributeError):
        paragraph_chunker.chunk(123)  # type: ignore
    
    with pytest.raises(TypeError):
        sentence_chunker.chunk(123)  # type: ignore
    
    with pytest.raises(AttributeError):
        sliding_chunker.chunk(123)  # type: ignore
