"""Tests associated with Chunk module"""

import logging
from typing import List
import pytest

from RAGToolBox.chunk import (
    Chunker,
    ParagraphChunker,
    SentenceChunker,
    SlidingWindowChunker,
    SectionAwareChunker,
    HierarchicalChunker
)
from RAGToolBox.logging import RAGTBLogger, LoggingConfig

# Test data
SAMPLE_TEXT = (
    "This is the first paragraph. It contains multiple sentences. "
    "Each sentence should be properly handled.\n\n"
    "This is the second paragraph. It has different content."
    "The chunker should separate these paragraphs.\n\n"
    "This is the third paragraph. It's shorter than the others."
    )

SAMPLE_TEXT_NO_PARAGRAPHS = (
    "This is a single paragraph. It has multiple sentences. "
    "But no paragraph breaks."
    )

SAMPLE_TEXT_EMPTY_PARAGRAPHS = """First paragraph.

Second paragraph.

Third paragraph."""

SAMPLE_TEXT_MIXED_WHITESPACE = """   First paragraph with leading spaces.


Second paragraph with extra newlines.


Third paragraph with trailing spaces.   """

# Section-aware chunker test data
SAMPLE_MARKDOWN_TEXT = """# Main Title

This is the introduction paragraph. It contains some basic information.

## Section 1

This is the first section. It has multiple sentences. Each sentence provides different information.

This is another paragraph in section 1. It continues the discussion.

## Section 2

This is the second section. It's shorter than the first section.

### Subsection 2.1

This is a subsection. It's nested under section 2## Section 3

This is the final section. It concludes the document."""

SAMPLE_MARKDOWN_NO_HEADERS = """This is a document without any markdown headers.

It just has regular paragraphs. No special formatting."""

SAMPLE_MARKDOWN_EMPTY_SECTIONS = """# Title

## Section 1

## Section 2

Some content here.

## Section 3
"""

SAMPLE_MARKDOWN_LARGE_SECTION = (
    "# Large Document\n\n"
    "## Introduction\n\n"
    "This is a very long section that will need to be split into multiple chunks. "
    "Lorem ipsum dolor sit amet. " * 50 +"\n\n"
    "## Conclusion\n\n"
    "This is the end."
    )

# =====================
# UNIT TESTS
# =====================

# SectionAwareChunker unit tests
def test_section_aware_chunker_basic() -> None:
    """Test section_aware chunker handles markdown patterns"""
    chunker = SectionAwareChunker(max_chunk_size=200, overlap=50)
    chunks = chunker.chunk(SAMPLE_MARKDOWN_TEXT)
    assert len(chunks) >= 1
    assert any("# Main Title" in chunk for chunk in chunks)
    assert any("## Section 1" in chunk for chunk in chunks)
    assert any("## Section 2" in chunk for chunk in chunks)
    assert any("## Section 3" in chunk for chunk in chunks)

def test_section_aware_chunker_no_headers() -> None:
    """Test section_aware chunker can handle missing headers"""
    chunker = SectionAwareChunker(max_chunk_size=200, overlap=50)
    chunks = chunker.chunk(SAMPLE_MARKDOWN_NO_HEADERS)
    assert len(chunks) >= 1
    # Should still chunk the text even without headers
    assert any("document without any markdown headers" in chunk for chunk in chunks)

def test_section_aware_chunker_empty_sections() -> None:
    """Test section_aware chunker preserves headers for empty sections"""
    chunker = SectionAwareChunker(max_chunk_size=200, overlap=50)
    chunks = chunker.chunk(SAMPLE_MARKDOWN_EMPTY_SECTIONS)
    assert len(chunks) >= 1
    # Should preserve headers even for empty sections
    assert any("# Title" in chunk for chunk in chunks)
    assert any("## Section 1" in chunk for chunk in chunks)
    assert any("## Section 2" in chunk for chunk in chunks)
    assert any("## Section 3" in chunk for chunk in chunks)

def test_section_aware_chunker_large_section() -> None:
    """Test section_aware chunker splits large sections into smaller sections"""
    chunker = SectionAwareChunker(max_chunk_size=500, overlap=100)
    chunks = chunker.chunk(SAMPLE_MARKDOWN_LARGE_SECTION)
    assert len(chunks) >=2
    # Should split the large section
    assert any("# Large Document" in chunk for chunk in chunks)
    assert any("## Introduction" in chunk for chunk in chunks)
    assert any("## Conclusion" in chunk for chunk in chunks)

def test_section_aware_chunker_empty_text() -> None:
    """Test to check section_aware chunker handles empty strings"""
    chunker = SectionAwareChunker()
    chunks = chunker.chunk("")
    assert len(chunks) == 0

def test_section_aware_chunker_whitespace_only() -> None:
    """Test to check section_aware chunker doesn't capture whitespace"""
    chunker = SectionAwareChunker()
    chunks = chunker.chunk("   \n\n   \n\n   ")
    assert len(chunks) == 0

def test_section_aware_chunker_single_header() -> None:
    """
    Test to check section_aware chunker handles a single header and
    doesn't split if within max_chunk_size
    """
    text = "# Single Header\n\nSome content here."
    chunker = SectionAwareChunker(max_chunk_size=100, overlap=20)
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert "# Single Header" in chunks[0]
    assert "Some content here" in chunks[0]

def test_section_aware_chunker_nested_headers() -> None:
    """Test to check section_aware chunker handles nested headers"""
    text = (
        "# Title\n\n## Section 1\n\n### Subsection11\n"
        "Content here.\n\n### Subsection12\nMore content.\n\n"
        "## Section 2\n\nFinal content."
        )
    chunker = SectionAwareChunker(max_chunk_size=200, overlap=50)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1  # Check that all header levels are preserved
    assert any("# Title" in chunk for chunk in chunks)
    assert any("## Section 1" in chunk for chunk in chunks)
    assert any("### Subsection11" in chunk for chunk in chunks)
    assert any("### Subsection12" in chunk for chunk in chunks)
    assert any("## Section 2" in chunk for chunk in chunks)

def test_section_aware_chunker_break_point_selection() -> None:
    """Test that section_aware chunker finds good break points"""
    text = (
        "# Test Document\n\n## Long Section\nThis is a sentence. " * 5 + "\n\n"
        "## Short Section\nBrief content."
        )
    chunker = SectionAwareChunker(max_chunk_size=300, overlap=50)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    # Should split the long section
    # Check that chunks don't break in the middle of words
    for chunk in chunks:
        if len(chunk) > 50:  # Skip very short chunks
            # Should not end with a partial word
            assert not chunk.strip().endswith('T')
            assert not chunk.strip().endswith('Th')
            assert not chunk.strip().endswith('Thi')

def test_section_aware_chunker_parameter_validation(
    caplog: pytest.LogCaptureFixture
    ) -> None:
    """Test invalid parameters are handled for section_aware chunker"""
    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    err_1 = "'max_chunk_size' must be positive when using 'SectionAwareChunker'"
    err_2 = "'overlap' must be non-negative when using 'SectionAwareChunker'"
    err_3 = "'overlap' must be less than 'max_chunk_size' when using 'SectionAwareChunker'"

    with pytest.raises(ValueError, match=err_1):
        SectionAwareChunker(max_chunk_size=-10)
    with pytest.raises(ValueError, match=err_2):
        SectionAwareChunker(overlap=-1)
    with pytest.raises(ValueError, match=err_3):
        SectionAwareChunker(max_chunk_size=10, overlap=100)
    assert err_1 in caplog.text
    assert err_2 in caplog.text
    assert err_3 in caplog.text

def test_section_aware_chunker_protocol_compliance() -> None:
    """Test output dtype compliance of section_aware chunker"""
    chunker = SectionAwareChunker()
    def process_with_chunker(chunker: Chunker, text: str) -> List[str]:
        return chunker.chunk(text)
    result = process_with_chunker(chunker, SAMPLE_MARKDOWN_TEXT)
    assert isinstance(result, list)

def test_section_aware_chunker_with_none_input() -> None:
    """Test section_aware chunker handles dtype: None input"""
    chunker = SectionAwareChunker()
    with pytest.raises(AttributeError):
        chunker.chunk(None)  # type: ignore

def test_section_aware_chunker_with_non_string_input() -> None:
    """Test section_aware chunker handles non-string dtypes"""
    chunker = SectionAwareChunker()
    with pytest.raises(AttributeError):
        chunker.chunk(123)  # type: ignore

def test_section_aware_chunker_no_word_clipping() -> None:
    """
    Test section_aware chunker creates a long section with
    a word that would be split by naive chunking
    """
    header = "## Section"
    text = header + "\n" + (
        "This is a long section with a wordthatshouldnotbesplit " * 10
    )
    chunker = SectionAwareChunker(max_chunk_size=80, overlap=10)
    chunks = chunker.chunk(text)
    # Check that the full word is present in at least one chunk
    assert any('wordthatshouldnotbesplit' in chunk for chunk in chunks)

# ParagraphChunker unit tests
def test_paragraph_chunker_basic() -> None:
    """Basic smokescreen test for paragraph chunker"""
    chunkers = [
        ParagraphChunker(),
        SlidingWindowChunker(window_size=1000, overlap=200),
        HierarchicalChunker([ParagraphChunker()]),
    ]
    for chunker in chunkers:
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) >= 1
        if isinstance(chunker, (ParagraphChunker, HierarchicalChunker)):
            assert any("first paragraph" in c for c in chunks)
            assert any("second paragraph" in c for c in chunks)
            assert any("third paragraph" in c for c in chunks)
        else:
            # For window-based chunkers, just check the phrases are present somewhere
            assert any("first paragraph" in c for c in chunks)
            assert any("second paragraph" in c for c in chunks)
            assert any("third paragraph" in c for c in chunks)

def test_paragraph_chunker_single_paragraph() -> None:
    """Test for a single paragraph input to paragraph chunker"""
    chunkers = [
        ParagraphChunker(),
        SlidingWindowChunker(window_size=1000, overlap=200),
        HierarchicalChunker([ParagraphChunker()]),
    ]
    for chunker in chunkers:
        chunks = chunker.chunk(SAMPLE_TEXT_NO_PARAGRAPHS)
        assert len(chunks) >= 1
        if isinstance(chunker, (ParagraphChunker, HierarchicalChunker)):
            assert any("single paragraph" in c for c in chunks)
        else:
            assert any("single paragraph" in c for c in chunks)

def test_paragraph_chunker_empty_paragraphs() -> None:
    """Test paragraph chunker on empty paragraph"""
    chunkers = [
        ParagraphChunker(),
        SlidingWindowChunker(window_size=1000, overlap=200),
        HierarchicalChunker([ParagraphChunker()]),
    ]
    for chunker in chunkers:
        chunks = chunker.chunk(SAMPLE_TEXT_EMPTY_PARAGRAPHS)
        assert len(chunks) >= 1
        # All chunks should be non-empty
        assert all(chunk.strip() for chunk in chunks)

def test_paragraph_chunker_mixed_whitespace() -> None:
    """Test paragraph chunker handles mixed whitespace"""
    chunkers = [
        ParagraphChunker(),
        SlidingWindowChunker(window_size=1000, overlap=200),
        HierarchicalChunker([ParagraphChunker()]),
    ]
    for chunker in chunkers:
        chunks = chunker.chunk(SAMPLE_TEXT_MIXED_WHITESPACE)
        assert len(chunks) >= 1
        if isinstance(chunker, (ParagraphChunker, HierarchicalChunker)):
            assert any(chunk.startswith("First paragraph") for chunk in chunks)
            assert any(chunk.startswith("Second paragraph") for chunk in chunks)
            assert any(chunk.startswith("Third paragraph") for chunk in chunks)
        else:
            # For window-based chunkers, just check the phrases are present somewhere
            assert any("First paragraph" in chunk for chunk in chunks)
            assert any("Second paragraph" in chunk for chunk in chunks)
            assert any("Third paragraph" in chunk for chunk in chunks)

def test_paragraph_chunker_empty_text() -> None:
    """Test paragraph chunker handles empty text"""
    chunkers = [
        ParagraphChunker(),
        SlidingWindowChunker(window_size=1000, overlap=200),
        HierarchicalChunker([ParagraphChunker()]),
    ]
    for chunker in chunkers:
        chunks = chunker.chunk("")
        assert len(chunks) == 0

def test_paragraph_chunker_whitespace_only() -> None:
    """Test paragraph chunker handles whitespace only"""
    chunkers = [
        ParagraphChunker(),
        SlidingWindowChunker(window_size=1000, overlap=200),
        HierarchicalChunker([ParagraphChunker()]),
    ]
    for chunker in chunkers:
        chunks = chunker.chunk("   \n\n   \n\n   ")
        assert len(chunks) == 0

# SentenceChunker unit tests
def test_sentence_chunker_basic() -> None:
    """Basic smokescreen test for sentence chunker"""
    chunkers = [
        SentenceChunker(),
        HierarchicalChunker([SentenceChunker()]),
    ]
    for chunker in chunkers:
        chunks = chunker.chunk(SAMPLE_TEXT_NO_PARAGRAPHS)
        assert len(chunks) == 3
        assert "single paragraph" in chunks[0]
        assert "multiple sentences" in chunks[1]
        assert "paragraph breaks" in chunks[2]

def test_sentence_chunker_complex_sentences() -> None:
    """Test sentence chunker can handle sentences with literal dot chars"""
    complex_text = (
        "Dr. Smith said: 'This is a quote.' "
        "Mr. Jones replied. What about abbreviations like etc.?"
        )
    chunkers = [
        SentenceChunker(),
        HierarchicalChunker([SentenceChunker()]),
    ]
    for chunker in chunkers:
        chunks = chunker.chunk(complex_text)
        assert len(chunks) == 4
        assert "Dr. Smith said" in chunks[0]
        assert "Mr. Jones replied" in chunks[1]
        assert "abbreviations like etc" in chunks[2]
        assert chunks[3] == "?"

def test_sentence_chunker_empty_text() -> None:
    """Test sentence chunker can handle empty strings"""
    chunkers = [
        SentenceChunker(),
        HierarchicalChunker([SentenceChunker()]),
    ]
    for chunker in chunkers:
        chunks = chunker.chunk("")
        assert len(chunks) == 0

def test_sentence_chunker_single_sentence() -> None:
    """Test sentence chunker doesn't split single sentence"""
    chunkers = [
        SentenceChunker(),
        HierarchicalChunker([SentenceChunker()]),
    ]
    for chunker in chunkers:
        chunks = chunker.chunk("This is a single sentence.")
        assert len(chunks) == 1
        assert chunks[0] == "This is a single sentence."

def test_sentence_chunker_with_numbers() -> None:
    """Test sentence chunker handles floating point number strings"""
    text = "The study used 1.5 Tesla MRI. Fig. 1 shows the results. The p-value was < 0.05."
    chunkers = [
        SentenceChunker(),
        HierarchicalChunker([SentenceChunker()]),
    ]
    for chunker in chunkers:
        chunks = chunker.chunk(text)
        assert len(chunks) == 4
        assert "1.5 Tesla MRI" in chunks[0]
        assert "Fig." in chunks[1]
        assert "1 shows the results" in chunks[2]
        assert "p-value was < 0.05" in chunks[3]

def test_sentence_chunker_with_quotes() -> None:
    """Test sentence chunker handles quotes"""
    text = 'He said "This is quoted text." Then he continued. "Another quote."'
    chunkers = [
        SentenceChunker(),
        HierarchicalChunker([SentenceChunker()]),
    ]
    for chunker in chunkers:
        chunks = chunker.chunk(text)
        assert len(chunks) == 3
        assert "quoted text" in chunks[0]
        assert "he continued" in chunks[1]
        assert "Another quote" in chunks[2]

# Protocol compliance and error handling
def test_chunker_protocol_compliance() -> None:
    """Basic protocol compliance test with chunkers"""
    chunkers = [
        ParagraphChunker(),
        SentenceChunker(),
        SlidingWindowChunker(),
        SectionAwareChunker(),
        HierarchicalChunker([ParagraphChunker(), SentenceChunker()])
    ]
    for chunker in chunkers:
        def process_with_chunker(chunker: Chunker, text: str) -> List[str]:
            return chunker.chunk(text)
        result = process_with_chunker(chunker, "Test text.")
        assert isinstance(result, list)

def test_chunker_with_none_input() -> None:
    """Test chunkers handle dtype: None inputs"""
    chunkers = [
        ParagraphChunker(),
        SentenceChunker(),
        SlidingWindowChunker(),
        SectionAwareChunker(),
        HierarchicalChunker([ParagraphChunker()])
    ]
    for chunker in chunkers:
        if isinstance(chunker, SentenceChunker):
            with pytest.raises(TypeError):
                chunker.chunk(None)  # type: ignore
        else:
            with pytest.raises(AttributeError):
                chunker.chunk(None)  # type: ignore

def test_chunker_with_non_string_input() -> None:
    """Test chunkers handle non-string inputs"""
    chunkers = [
        ParagraphChunker(),
        SentenceChunker(),
        SlidingWindowChunker(),
        SectionAwareChunker(),
        HierarchicalChunker([ParagraphChunker()])
    ]
    for chunker in chunkers:
        if isinstance(chunker, SentenceChunker):
            with pytest.raises(TypeError):
                chunker.chunk(123)  # type: ignore
        else:
            with pytest.raises(AttributeError):
                chunker.chunk(123)  # type: ignore

# SlidingWindowChunker unit tests
def test_sliding_window_chunker_no_word_clipping() -> None:
    """Test sliding_window_chunker doesn't clip words at window boundaries"""
    # The word 'boundaryword' is placed so it would be clipped by a naive window
    text = (
        "This is a test sentence that is designed to end right at the window "
        "boundaryword and then continues with more text after the boundaryword."
    )
    # Set window_size so that 'boundaryword' would be split if not careful
    window_size = text.find('boundaryword') + 5  # Intentionally split in the middle
    chunker = SlidingWindowChunker(window_size=window_size, overlap=10)
    chunks = chunker.chunk(text)
    # Assert that 'boundaryword' is never split across chunks
    for chunk in chunks:
        assert 'boundaryword' in chunk or 'boundaryword.' in chunk
        # Check that chunk boundaries are at whitespace or start/end of text
        if chunk:
            start_idx = text.find(chunk)
            end_idx = start_idx + len(chunk)
            if start_idx > 0:
                assert (
                    text[start_idx - 1] in (' ', '\n')
                    ), (
                    f"Chunk does not start at whitespace: {chunk!r}"
                    )
            if end_idx < len(text):
                assert (
                    text[end_idx] in (' ', '\n', '.')
                    ), (
                    f"Chunk does not end at whitespace: {chunk!r}"
                    )
    # Also check that the full word is present in at least one chunk
    assert any('boundaryword' in chunk for chunk in chunks)

def test_sliding_window_chunker_input_errors_handled(caplog: pytest.LogCaptureFixture) -> None:
    """Test that SlidingWindowChunker inialization handles input errors correctly"""
    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    err_1 = "'window_size' must be positive when using 'SlidingWindowChunker'"
    err_2 = "'overlap' must be non-negative when using 'SlidingWindowChunker'"
    err_3 = "'overlap' must be less than 'window_size' when using 'SlidingWindowChunker'"

    with pytest.raises(ValueError, match=err_1):
        SlidingWindowChunker(window_size=-10)
    with pytest.raises(ValueError, match=err_2):
        SlidingWindowChunker(overlap=-1)
    with pytest.raises(ValueError, match=err_3):
        SlidingWindowChunker(window_size=10, overlap=100)
    assert err_1 in caplog.text
    assert err_2 in caplog.text
    assert err_3 in caplog.text

def test_hierarchical_chunker_input_errors_handled(caplog: pytest.LogCaptureFixture) -> None:
    """Test that HierarchicalChunker inialization handles input errors correctly"""
    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    err_msg = "At least one chunker must be provided when using 'HierarchicalChunker'"
    with pytest.raises(ValueError, match=err_msg):
        HierarchicalChunker(chunkers=[])
    assert err_msg in caplog.text

# =====================
# INTEGRATION TESTS
# =====================

def test_section_aware_chunker_with_real_document_content() -> None:
    """Full test of section_aware chunker class with document"""
    real_document = """# A review of low-intensity focused ultrasound for neuromodulation

## Abstract

Low-intensity focused ultrasound (LIFU) is a non-invasive neuromodulation technique that has shown promise for various clinical applications. This review discusses the mechanisms, applications, and future directions of LIFU.

## Introduction

LIFU works by focusing ultrasound waves into small regions of the brain. The technique offers several advantages over traditional neuromodulation methods.

## Methods

Studies were conducted using various LIFU parameters. Results showed significant effects on neural activity.

## Results

LIFU successfully modulated neural activity in target regions. No adverse effects were observed.

## Discussion

These findings suggest LIFU has potential for clinical applications. Further research is needed to optimize parameters.

## Conclusion

LIFU represents a promising new approach to neuromodulation."""
    chunker = SectionAwareChunker(max_chunk_size=80, overlap=10)
    chunks = chunker.chunk(real_document)
    assert len(chunks) >= 3  # Check that all major sections are represented
    assert any("## Abstract" in chunk for chunk in chunks)
    assert any("## Introduction" in chunk for chunk in chunks)
    assert any("## Methods" in chunk for chunk in chunks)
    assert any("## Results" in chunk for chunk in chunks)
    assert any("## Discussion" in chunk for chunk in chunks)
    assert any("## Conclusion" in chunk for chunk in chunks)

def test_section_aware_chunker_with_ncbi_loader_output() -> None:
    """Test section_aware chunker with NCBILoader test output"""
    # Simulate output from NCBILoader
    ncbi_output = (
        "# A review of low-intensity focused ultrasound for neuromodulation\n"
        "**Authors:** Hongchae Baek, Ki Joo Pahk, Hyungmin Kim\n"
        "**Journal:** Biomedical Engineering Letters\n"
        "**DOI:** 10.107/s1353407*Keywords:** Focused ultrasound, "
        "Neuromodulation, Brain, Non-invasive\n"
        "---\n"
        "## Abstract\n"
        "Abstracts The ability of ultrasound to be focused into a small region of interest "
        "through the intact skull within the brain has led researchers to investigate its "
        "potential therapeutic uses for functional neurosurgery and tumor ablation. "
        "Studies have used high-intensity focused ultrasound to ablate tissue in localised "
        "brain regions for movement disorders and chronic pain while sparing the overlying and "
        "surrounding tissue. More recently, low-intensity focused ultrasound (LIFU) that induces "
        "reversible biological effects has been emerged as an alternative neuromodulation modality "
        "due to its bi-modal ( i.e. excitation and suppression) capability with exquisite spatial "
        "specificity and depth penetration. Many compelling evidences of LIFU-mediated "
        "neuromodulatory effects including behavioral responses, electrophysiological recordings "
        "and functional imaging data have been found in the last decades. LIFU, therefore, has "
        "the enormous potential to improve the clinical outcomes as well as to replace the "
        "currently available neuromodulation techniques such as deep brain stimulation (DBS), "
        "transcranial magnetic stimulation and transcranial current stimulation. In this paper, "
        "we aim to provide a summary of pioneering studies in the field of ultrasonic "
        "neuromodulation including its underlying mechanisms that were published in the last 60 "
        "years. In closing, some of potential clinical applications of ultrasonic brain "
        "stimulation will be discussed."
        )
    chunker = SectionAwareChunker(max_chunk_size=400, overlap=100)
    chunks = chunker.chunk(ncbi_output)

    assert len(chunks) >= 2
    # Check that metadata and content are preserved
    assert any("# A review of low-intensity focused ultrasound" in chunk for chunk in chunks)
    assert any("**Authors:** Hongchae Baek, Ki Joo Pahk, Hyungmin Kim" in chunk for chunk in chunks)
    assert any("## Abstract" in chunk for chunk in chunks)
    assert any("LIFU-mediated neuromodulatory effects" in chunk for chunk in chunks)

def test_chunker_performance_with_large_text() -> None:
    """Test chunker handles large text documents"""
    # Test performance with larger text
    large_text = "# Large Document\n\n" + "This is a sentence. " * 10

    chunkers = [
        ParagraphChunker(),
        SentenceChunker(),
        SlidingWindowChunker(window_size=1000, overlap=200),
        SectionAwareChunker(max_chunk_size=100, overlap=50)
    ]

    for chunker in chunkers:
        chunks = chunker.chunk(large_text)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        # All chunks should be non-empty
        assert all(chunk.strip() for chunk in chunks)

def test_chunker_edge_cases() -> None:
    """Edge case tests for chunkers"""
    edge_cases = [
        "",  # Empty string
        "   \n\n   \n\n  ", # Whitespace only
        "# Header only",  # Header without content
        "Content only",  # Content without header
        "# H12\n### H3\nContent",  # Nested headers
        "Header\n\n\nContent",  # Multiple newlines
    ]

    chunkers = [
        ParagraphChunker(),
        SentenceChunker(),
        SlidingWindowChunker(),
        SectionAwareChunker(),
    ]

    for chunker in chunkers:
        for text in edge_cases:
            chunks = chunker.chunk(text)
            assert isinstance(chunks, list)
            # All chunks should be non-empty (except for empty input)
            if text.strip():
                assert all(chunk.strip() for chunk in chunks)

def test_chunker_consistency() -> None:
    """Test to see chunking is not stochastic"""
    # Test that chunkers produce consistent results
    text = SAMPLE_MARKDOWN_TEXT

    chunkers = [
        ParagraphChunker(),
        SentenceChunker(),
        SlidingWindowChunker(window_size=500),
        SectionAwareChunker(max_chunk_size=50, overlap=25)
    ]

    for chunker in chunkers:
        # Run multiple times to ensure consistency
        chunks1 = chunker.chunk(text)
        chunks2 = chunker.chunk(text)
        assert len(chunks1) == len(chunks2)
        assert chunks1 == chunks2
