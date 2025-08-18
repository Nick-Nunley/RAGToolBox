"""Tests associated with Loader module"""
# pylint: disable=protected-access

import logging
from typing import Any
from unittest.mock import Mock
from pathlib import Path
import pytest
import requests
import pdfplumber
from Bio import Entrez

from RAGToolBox.loader import (
    BaseLoader,
    NCBILoader,
    TextLoader,
    HTMLLoader,
    PDFLoader,
    UnknownLoader
    )

from RAGToolBox.logging import RAGTBLogger, LoggingConfig

# Helpers and Mocks
class DummyPage:
    """Mock page class"""
    def __init__(self, text: str) -> None:
        self._text = text
    def extract_text(self) -> str:
        """Method to return page 'content'"""
        return self._text

class DummyPDF:
    """Mock PDF class"""
    def __init__(self, pages: list) -> None:
        self.pages = pages
    def __enter__(self) -> 'DummyPDF':
        return self
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return False

class DummyResponse:
    """Mock response"""
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code
    def raise_for_status(self) -> None:
        """Method to mock a raise status error"""
        if self.status_code != 200:
            raise requests.HTTPError(f"Status code: {self.status_code}")

class DummyHandle:
    """Mock page handles"""
    def __init__(self, text: bytes) -> None:
        self._text = text
    def read(self) -> bytes:
        """Method to read bytes"""
        return self._text
    def close(self) -> None:
        """Method to 'close' file"""


# =====================
# UNIT TESTS
# =====================

def test_detect_loader_txt() -> None:
    """Test detect_loader detects .txt extension"""
    cls = BaseLoader.detect_loader('http://example.com/file.txt', b'hello')
    assert cls is TextLoader

def test_detect_loader_md() -> None:
    """Test detect_loader detects .md extension"""
    cls = BaseLoader.detect_loader('http://example.com/file.md', b'')
    assert cls is TextLoader

def test_detect_loader_html() -> None:
    """Test detect_loader detects .html extension"""
    cls = BaseLoader.detect_loader('http://example.com/index.html', b'<html>')
    assert cls is HTMLLoader

def test_detect_loader_pdf() -> None:
    """Test detect_loader detects .pdf extension"""
    cls = BaseLoader.detect_loader('http://example.com/doc.pdf', b'%PDF')
    assert cls is PDFLoader

def test_detect_loader_unknown() -> None:
    """Test detect_loader passes unknown file formats to UnknownLoader"""
    cls = BaseLoader.detect_loader('http://example.com/archive.zip', b'PK')
    assert cls is UnknownLoader

def test_detect_loader_ncbi() -> None:
    """Test detect_loader detects a PMC URL"""
    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC123456/'
    cls = BaseLoader.detect_loader(url, b'ignored')
    assert cls is NCBILoader

def test_detect_loader_local_txt(tmp_path: Path) -> None:
    """Test detect_loader handles a local .txt filepath"""
    p = tmp_path / "file.txt"
    p.write_text("hello", encoding="utf-8")
    cls = BaseLoader.detect_loader(str(p), b"ignored")
    assert cls is TextLoader

def test_detect_loader_local_pdf(tmp_path: Path) -> None:
    """Test detect_loader handles a local .pdf filepath"""
    p = tmp_path / "document.pdf"
    p.write_bytes(b"%PDF-1.4 fake pdf")
    cls = BaseLoader.detect_loader(str(p), b"ignored")
    assert cls is PDFLoader

def test_detect_loader_local_html(tmp_path: Path) -> None:
    """Test detect_loader handles a local .html filepath"""
    p = tmp_path / "page.html"
    p.write_text("<html><body>content</body></html>", encoding="utf-8")
    cls = BaseLoader.detect_loader(str(p), b"ignored")
    assert cls is HTMLLoader

def test_detect_loader_local_unknown_html_by_content(tmp_path: Path) -> None:
    """Local file with unknown extension should be detected as HTML via content sniffing."""
    p = tmp_path / "blob.bin"
    p.write_bytes(b"irrelevant")
    sniff = b"  \n\t<!DoCtYpe hTmL><html><body>hi</body></html>"
    cls = BaseLoader.detect_loader(str(p), sniff)
    assert cls is HTMLLoader

def test_detect_loader_local_unknown_pdf_by_content(tmp_path: Path) -> None:
    """Test detect_loader handles unknown local file and falls back to sniffing"""
    p = tmp_path / "blob.bin"
    p.write_bytes(b"")
    cls = BaseLoader.detect_loader(str(p), b"%PDF")  # content sniff should pick PDF
    assert cls is PDFLoader

def test_handle_local_detection_unknown_fallback_logs_warning(
    caplog: pytest.LogCaptureFixture
    ) -> None:
    """Test that _handle_local_detection with unknown logs fallback"""
    caplog.set_level(logging.WARNING)
    # Neither extension nor content looks recognizable
    cls = BaseLoader._handle_local_file_detection("data.bin", b"\x00\x01\x02\x03binary")
    assert cls is TextLoader

    # Ensure we warned and mentioned fallback to TextLoader
    assert any(
        "Unknown format detected while loading" in rec.getMessage()
        and "Falling back to TextLoader" in rec.getMessage()
        for rec in caplog.records
    )

def test_base_loader_fetch_error_handling(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
    """Test BaseLoader throws proper fetching errors"""
    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))
    test_loader = BaseLoader(source = '', output_dir = '')

    def dummy_get(url: str, timeout: float) -> None: # pylint: disable=redefined-builtin
        """Mock fetch function to return timeout error"""
        raise requests.Timeout("simulated timeout")
    monkeypatch.setattr("RAGToolBox.loader.requests.get", dummy_get)

    with pytest.raises(TimeoutError) as exc:
        test_loader.fetch()
    err_msg = 'Timed out after'
    assert err_msg in str(exc.value)
    assert err_msg in caplog.text

    fake_response = Mock()
    def fake_raise():
        raise requests.HTTPError("500 Server Error")
    fake_response.raise_for_status = fake_raise
    monkeypatch.setattr("RAGToolBox.loader.requests.get", lambda url, timeout: fake_response)

    with pytest.raises(RuntimeError) as exc:
        test_loader.fetch()
    err_msg = 'Error fetching'
    assert err_msg in str(exc.value)
    assert err_msg in caplog.text

def test_ncbi_loader_with_malform_source(
    caplog: pytest.LogCaptureFixture
    ) -> None:
    """Test that NCBILoader._check_available_sources errors on malformed source"""
    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    loader = NCBILoader("https://pmc.ncbi.nlm.nih.gov/articles/PMC123456/", "out")

    with pytest.raises(ValueError) as exc:
        loader._check_available_sources('bad_source')
    err = "is not supported by NCBILoader. See available sources"
    assert err in caplog.text
    assert err in str(exc.value)

def test_ncbi_loader_fetch_success(monkeypatch: Any) -> None:
    """Test NCBILoader fetches properly"""
    calls = []
    def dummy_efetch(
        db: str, id: str, rettype: str, retmode: str # pylint: disable=redefined-builtin
        ) -> DummyHandle:
        """Mock fetch function for Entrez.efetch"""
        calls.append((db, id, rettype, retmode))
        # Minimal valid PMC XML with an abstract
        xml = b'<?xml version="1.0"?><article><abstract>Abstract text</abstract></article>'
        return DummyHandle(xml)
    monkeypatch.setattr(Entrez, 'efetch', dummy_efetch)

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC999999/'
    loader = NCBILoader(url, 'out')
    loader.fetch()
    assert loader.raw_content is not None
    assert b'Abstract text' in loader.raw_content # type: ignore[union-attr]

    loader.convert()
    assert 'Abstract text' in loader.text
    assert calls[0] == ('pmc', 'PMC999999', 'full', 'xml')

def test_ncbi_loader_fetch_abstract_only(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
    """Test NCBILoader fetch logs abstract only warning"""
    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    calls = []
    def dummy_efetch(
        db: str, id: str, rettype: str, retmode: str # pylint: disable=redefined-builtin
        ) -> DummyHandle:
        """Mock fetch function for Entrez.efetch"""
        calls.append((db, id, rettype, retmode))
        # Minimal valid PMC XML with an abstract
        xml = b"""<?xml version="1.0"?>
        <article>
        <front>
            <notes>Publisher note: This journal does not allow downloading of the full text.</notes>
        </front>
        <abstract>Abstract text</abstract>
        </article>"""
        return DummyHandle(xml)
    monkeypatch.setattr(Entrez, 'efetch', dummy_efetch)

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC999999/'
    loader = NCBILoader(url, 'out')
    loader.fetch()
    assert loader.raw_content is not None
    assert b'Abstract text' in loader.raw_content # type: ignore[union-attr]
    assert 'Warning: Full text not available for' in caplog.text

    loader.convert()
    assert 'Abstract text' in loader.text
    assert calls[0] == ('pmc', 'PMC999999', 'full', 'xml')

def test_ncbi_loader_fetch_failure(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
    """Test NCBILoader raises RuntimeError for broken PMC URL"""
    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))
    def dummy_efetch(
        db: str, id: str, rettype: str, retmode: str # pylint: disable=redefined-builtin
        ) -> None:
        """Mock fetch function to return down status"""
        raise Exception('NCBI down') # pylint: disable=broad-exception-raised
    monkeypatch.setattr(Entrez, 'efetch', dummy_efetch)

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC000000/'
    loader = NCBILoader(url, 'out')
    with pytest.raises(RuntimeError) as exc:
        loader.fetch()
    err_msg = 'Entrez fetch failed for PMC000000'
    assert err_msg in str(exc.value)
    assert err_msg in caplog.text

def test_ncbi_loader_fetch_with_nonstandard_link(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
    """Test NCBILoader warns about not finding PMC or PubMed in link"""
    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    calls = []
    def dummy_efetch(
        db: str, id: str, rettype: str, retmode: str # pylint: disable=redefined-builtin
        ) -> DummyHandle:
        """Mock fetch function for Entrez.efetch"""
        calls.append((db, id, rettype, retmode))
        xml = b'<?xml version="1.0"?><article><abstract>Abstract text</abstract></article>'
        return DummyHandle(xml)
    monkeypatch.setattr(Entrez, 'efetch', dummy_efetch)

    url = 'https://test.gov/articles/PMC000000/'
    loader = NCBILoader(url, 'out')
    loader.fetch()
    assert "Detected NCBI format, but could not find 'pmc.' or 'pubmed.' strings" in caplog.text
    assert loader.raw_content is not None
    assert b'Abstract text' in loader.raw_content # type: ignore[union-attr]

    loader.convert()
    assert 'Abstract text' in loader.text
    assert calls[0] == ('pmc', 'PMC000000', 'full', 'xml')

def test_ncbi_loader_fetch_pdf_success(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
    """Test NCBI loader extracts PDF links"""
    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))
    # Minimal PMC XML with a relative PDF self-uri to exercise URL building
    xml = b"""<?xml version="1.0"?>
    <article xmlns:xlink="http://www.w3.org/1999/xlink">
      <front>
        <article-meta>
          <self-uri content-type="pmc-pdf" xlink:href="paper.pdf"/>
        </article-meta>
      </front>
    </article>"""

    # 1) Entrez.efetch -> returns XML handle
    monkeypatch.setattr(
        Entrez, "efetch", lambda db, id, rettype, retmode: DummyHandle(xml) # type: ignore
        )

    # 2) requests.get for PDF download -> returns bytes that look like a PDF
    def mock_get(
        url: str, timeout: float # pylint: disable=redefined-builtin, unused-argument
        ) -> DummyResponse:
        assert url.endswith("/pdf/paper.pdf")
        return DummyResponse(b"%PDF-1.4 fake-pdf-bytes")
    monkeypatch.setattr("RAGToolBox.loader.requests.get", mock_get)

    # 3) pdfplumber.open -> return a dummy PDF with pages so convert() concatenates text
    def mock_pdf_open(_file_obj: Any):
        return DummyPDF([DummyPage("PDF Page 1"), DummyPage("PDF Page 2")])
    monkeypatch.setattr(pdfplumber, "open", mock_pdf_open)

    url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC424242/"
    loader = NCBILoader(url, str(tmp_path))
    loader.fetch()

    # Assertions
    assert loader._used_pdf is True
    assert "PDF Page 1" in loader.text
    assert "PDF Page 2" in loader.text
    assert "Attempting to download and extract text from PDF." in caplog.text

def test_ncbi_loader_fetch_pdf_failure_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
    """Test that PDF downloading is triggered but doesn't work and raises a warning"""
    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    xml = b"""<?xml version="1.0"?>
    <article xmlns:xlink="http://www.w3.org/1999/xlink">
      <front>
        <article-meta>
          <self-uri content-type="pmc-pdf" xlink:href="paper.pdf"/>
        </article-meta>
      </front>
    </article>"""

    # 1) Entrez.efetch -> returns XML handle
    monkeypatch.setattr(
        Entrez, "efetch", lambda db, id, rettype, retmode: DummyHandle(xml) # type: ignore
        )

    # 2) Make requests.get (used by _download_pdf) raise via raise_for_status
    class BadResponse:
        """Mock response class"""
        content = b""
        def raise_for_status(self):
            """Dummy method to raise HTTP error"""
            raise requests.HTTPError("simulated 500")
    monkeypatch.setattr("RAGToolBox.loader.requests.get", lambda url, timeout: BadResponse())

    url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC999000/"
    loader = NCBILoader(url, "out")

    loader.fetch()

    # Assertions: we fell back
    assert loader._used_pdf is False
    assert "Falling back to XML extraction" in caplog.text

def test_ncbi_extract_pdf_url_xml_parse_error(
    caplog: pytest.LogCaptureFixture
    ) -> None:
    """Test _extract_pdf_url_from_xml returns None and logs a warning on XML parse failure."""
    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    # Minimal NCBILoader instance; source only matters for logging
    loader = NCBILoader("https://pmc.ncbi.nlm.nih.gov/articles/PMC123456/", "out")

    # Malformed XML to trigger ET.fromstring(...) exception
    bad_xml = b"<article><self-uri content-type='pmc-pdf' href='file.pdf'></article"

    result = loader._extract_pdf_url_from_xml(bad_xml)

    assert result is None
    assert "Error parsing XML for PDF link. Returning dtype=None." in caplog.text

def test_ncbi_parse_xml_unknown_root(caplog: pytest.LogCaptureFixture) -> None:
    """Test _parse_xml_content returns {} and logs a warning for unknown root tag."""
    caplog.set_level(logging.WARNING)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    loader = NCBILoader("https://pmc.ncbi.nlm.nih.gov/articles/PMC123456/", "out")
    # Root tag doesn't match the recognized PMC/PubMed endings
    loader.raw_content = b"<weirdroot><data/></weirdroot>"

    result = loader._parse_xml_content()
    assert result == {}
    assert "Unkonwn XML root tag" in caplog.text

def test_ncbi_parse_xml_malformed_xml(caplog: pytest.LogCaptureFixture) -> None:
    """Test _parse_xml_content returns None and logs a warning when XML parsing fails."""
    caplog.set_level(logging.WARNING)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    loader = NCBILoader("https://pmc.ncbi.nlm.nih.gov/articles/PMC123456/", "out")
    # Malformed XML to force ET.fromstring(...) to raise
    loader.raw_content = b"<article><unclosed></article"

    result = loader._parse_xml_content()
    assert result is None
    assert "Error parsing XML content. Returning dtype=None." in caplog.text

def test_text_loader_convert() -> None:
    """Test TextLoader.convert handles simple strings"""
    tr = TextLoader('http://example.com/test.txt', 'out')
    tr.raw_content = 'Hello, World!'.encode('utf-8')
    tr.convert()
    assert tr.text == 'Hello, World!'

def test_html_loader_convert() -> None:
    """Test HTMLLoader extracts content out of HTML tags"""
    html = '<html><body><h1>Hi</h1><p>Para</p></body></html>'
    hr = HTMLLoader('http://example.com/index.html', 'out')
    hr.raw_content = html.encode('utf-8')
    hr.convert()
    # Should contain markdown headings and paragraph text
    assert 'Hi' in hr.text
    assert 'Para' in hr.text

def test_html_loader_convert_with_no_content(
    caplog: pytest.LogCaptureFixture
    ) -> None:
    """Test that HTMLLoader.convert warns and returns empty string with no content"""
    caplog.set_level(logging.WARNING)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    loader = HTMLLoader('https://test.com/index.html', 'out')
    loader.convert()

    assert loader.text == ''
    assert 'Warning: no content detected when calling HTMLLoader.convert().' in caplog.text

def test_html_loader_navigablestring_markdown() -> None:
    """Test HTMLLoader implements NavigableString properly"""
    html = b"""
    <html>
      <head><title>Test</title></head>
      <body>
        <h1>Header</h1>
        <p>This is a <b>test</b> paragraph with <i>inline</i> tags.</p>
        <ul>
          <li>Item 1</li>
          <li>Item 2</li>
          <li>Item 3</li>
          <li>Item 4</li>
          <li>Item 5</li>
        </ul>
        <h2>Subheader</h2>
        <p>Another paragraph to increase length.</p>
        <p>Yet another paragraph to ensure we exceed the fallback threshold.</p>
      </body>
    </html>
    """
    loader = HTMLLoader("http://example.com", ".")
    loader.raw_content = html
    loader.convert()
    output = loader.text

    # Should contain the markdown header
    assert "# Header" in output or "# Test" in output
    assert "## Subheader" in output

    # Should contain the paragraph text, flattened
    assert "This is a test paragraph with inline tags." in output
    assert "Another paragraph to increase length." in output
    assert "Yet another paragraph to ensure we exceed the fallback threshold." in output

    # Should contain all list items as markdown
    for i in range(1, 6):
        assert f"- Item {i}" in output

def test_base_loader_convert_errors(caplog: pytest.LogCaptureFixture) -> None:
    """Test BaseLoader.convert throws error and subclasses inherit this"""
    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))
    test_loader = BaseLoader(source = '', output_dir = '')

    # Checking for error from BaseLoader
    err_msg = 'Subclasses must implement convert()'
    with pytest.raises(NotImplementedError, match=err_msg):
        test_loader.convert()
    assert err_msg in caplog.text

    # Checking subclasses inherit this
    class MockLoader(BaseLoader):
        """Dummy loader class"""
    test_loader = MockLoader(source = '', output_dir = '')

    with pytest.raises(NotImplementedError, match=err_msg):
        test_loader.convert()
    assert err_msg in caplog.text

def test_pdf_loader_convert(monkeypatch: Any) -> None:
    """Test PDFLoader.convert works"""
    # Mock pdfplumber.open to return DummyPDF with pages
    pages = [DummyPage('Page1'), DummyPage('Page2')]
    monkeypatch.setattr(pdfplumber, 'open', lambda _: DummyPDF(pages))
    pr = PDFLoader('http://example.com/doc.pdf', 'out')
    pr.raw_content = b'%PDF-1.4 dummy'
    pr.convert()
    assert 'Page1' in pr.text
    assert 'Page2' in pr.text

def test_pdf_loader_convert_with_no_content(
    caplog: pytest.LogCaptureFixture
    ) -> None:
    """Test that PDFLoader.convert warns and returns empty string with no content"""
    caplog.set_level(logging.WARNING)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    loader = PDFLoader('https://test.com/test.pdf', 'out')
    loader.convert()

    assert loader.text == ''
    assert 'Warning: no content detected when calling PDFLoader.convert().' in caplog.text

def test_text_loader_convert_with_no_content(
    caplog: pytest.LogCaptureFixture
    ) -> None:
    """Test that TextLoader.convert warns and returns empty string with no content"""
    caplog.set_level(logging.WARNING)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    loader = TextLoader('https://test.com/test.txt', 'out')
    loader.convert()

    assert loader.text == ''
    assert 'Warning: no content detected when calling TextLoader.convert().' in caplog.text

def test_unknown_loader_convert(caplog: pytest.LogCaptureFixture) -> None:
    """Test for UnknownLoader.convert method"""
    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))
    ur = UnknownLoader('http://example.com/file.xyz', 'out')
    ur.raw_content = b''
    ur.convert()
    assert 'Unknown format' in caplog.text
    assert ur.text == ''

def test_save(tmp_path: Any) -> None:
    """Test TextLoader.save method works"""
    tr = TextLoader('http://example.com/one.txt', tmp_path)
    tr.text = 'TestSave'
    tr.save()
    out_file = tmp_path / 'one.txt'
    assert out_file.exists()
    assert out_file.read_text(encoding='utf-8') == 'TestSave'

def test_html_loader_save_with_no_content(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
    """Test that HTMLLoader.save warns and saves empty file"""
    caplog.set_level(logging.WARNING)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    loader = HTMLLoader('https://test.com/index.html', tmp_path)
    loader.save()
    out = tmp_path / 'index.txt'

    assert out.exists()
    assert out.read_text(encoding='utf-8') == ''
    assert 'Warning: no content detected when calling HTMLLoader.save().' in caplog.text

# =====================
# INTEGRATION TESTS
# =====================

def test_full_process_text(monkeypatch: Any, tmp_path: Any) -> None:
    """Full test of loading process with TextLoader"""
    # Mock requests.get for a TXT URL
    monkeypatch.setattr(
        requests,
        'get',
        lambda url, timeout=None: DummyResponse(b'Some text content')
        )
    url = 'http://example.com/data.txt'

    LoaderClass = BaseLoader.detect_loader(url, b'some')
    loader = LoaderClass(url, str(tmp_path))
    loader.process()

    out_file = tmp_path / 'data.txt'
    assert out_file.exists()
    assert 'Some text content' in out_file.read_text(encoding='utf-8')

def test_full_process_html(monkeypatch: Any, tmp_path: Any) -> None:
    """Full test of loading process with HTMLLoader"""
    html = '<html><body><p>Hello HTML</p></body></html>'
    monkeypatch.setattr(
        requests,
        'get',
        lambda url, timeout=None: DummyResponse(html.encode('utf-8'))
        )
    url = 'http://example.com/page.html'

    LoaderClass = BaseLoader.detect_loader(url, html.encode('utf-8'))
    loader = LoaderClass(url, str(tmp_path))
    loader.process()

    out_file = tmp_path / 'page.txt'
    assert out_file.exists()
    assert 'Hello HTML' in out_file.read_text(encoding='utf-8')

def test_full_process_pdf(monkeypatch: Any, tmp_path: Any) -> None:
    """Full test of loading process with PDFLoader"""
    # Mock requests.get and pdfplumber.open
    monkeypatch.setattr(
        requests,
        'get',
        lambda url, timeout=None: DummyResponse(b'%PDF dummy content')
        )
    pages = [DummyPage('X'), DummyPage('Y')]
    monkeypatch.setattr(pdfplumber, 'open', lambda _: DummyPDF(pages))

    url = 'http://example.com/report.pdf'
    LoaderClass = BaseLoader.detect_loader(url, b'%PDF')
    loader = LoaderClass(url, str(tmp_path))
    loader.process()

    out_file = tmp_path / 'report.txt'
    assert out_file.exists()
    text = out_file.read_text(encoding='utf-8')
    assert 'X' in text
    assert 'Y' in text

def test_full_process_pmc(monkeypatch: Any, tmp_path: Any) -> None:
    """Full test of loading process with PMC article"""
    xml = b'<?xml version="1.0"?><article><abstract>Integration abstract text</abstract></article>'
    monkeypatch.setattr(Entrez, 'efetch', lambda db, id, rettype, retmode: DummyHandle(xml))
    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC111111/'
    LoaderClass = BaseLoader.detect_loader(url, b'ignored')
    loader = LoaderClass(url, str(tmp_path))
    loader.process()
    # Check that file was created and contains the abstract
    out_file = tmp_path / 'PMC111111.txt'
    assert out_file.exists()
    content = out_file.read_text(encoding='utf-8')
    assert 'Integration abstract text' in content

def test_full_process_pubmed(monkeypatch: Any, tmp_path: Any) -> None:
    """Full test of loading process with PubMed article"""
    xml = b'''<?xml version="1.0"?>
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <Article>
            <ArticleTitle>Test PubMed Article</ArticleTitle>
            <Abstract><AbstractText>PubMed abstract text</AbstractText></Abstract>
            <AuthorList>
              <Author>
                <LastName>Smith</LastName>
                <ForeName>Jane</ForeName>
              </Author>
            </AuthorList>
            <Journal><Title>Test Journal</Title></Journal>
            <ELocationID EIdType="doi">10.1234/testdoi</ELocationID>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>'''
    monkeypatch.setattr(Entrez, 'efetch', lambda db, id, rettype, retmode: DummyHandle(xml))
    url = 'https://pubmed.ncbi.nlm.nih.gov/123456/'
    LoaderClass = BaseLoader.detect_loader(url, b'ignored')
    loader = LoaderClass(url, str(tmp_path))
    loader.process()
    out_file = tmp_path / '123456.txt'
    assert out_file.exists()
    content = out_file.read_text(encoding='utf-8')
    assert 'Test PubMed Article' in content
    assert 'PubMed abstract text' in content
    assert 'Jane Smith' in content
    assert 'Test Journal' in content
    assert '10.1234/testdoi' in content

def test_full_process_local_file(tmp_path: Any) -> None:
    """Full test of loading process with local .txt file"""
    # Create a temporary text file
    test_file = tmp_path / 'test_document.txt'
    test_content = 'This is a test document with some content.'
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)

    # Process the local file
    loader = TextLoader(str(test_file), str(tmp_path / 'output'))
    loader.process()

    # Check that output file was created
    output_file = tmp_path / 'output' / 'test_document.txt'
    assert output_file.exists()

    # Check content
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
        assert content == test_content

def test_full_process_local_pdf(monkeypatch: Any, tmp_path: Any) -> None:
    """Full test of loading process with local PDF"""
    # Mock pdfplumber.open to return DummyPDF with pages
    def mock_pdf_open(file_obj: Any) -> DummyPDF: # pylint: disable=unused-argument
        """Mock PDF open function"""
        return DummyPDF([DummyPage("Page 1 content"), DummyPage("Page 2 content")])
    monkeypatch.setattr(pdfplumber, 'open', mock_pdf_open)

    # Create a temporary PDF file (content doesn't matter since we mock pdfplumber)
    test_file = tmp_path / 'test_document.pdf'
    with open(test_file, 'wb') as f:
        f.write(b'%PDF-1.4 fake pdf content')

    # Process the local file
    loader = PDFLoader(str(test_file), str(tmp_path / 'output'))
    loader.process()

    # Check that output file was created
    output_file = tmp_path / 'output' / 'test_document.txt'
    assert output_file.exists()

    # Check content
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
        assert 'Page 1 content' in content
        assert 'Page 2 content' in content

def test_full_process_with_no_text(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
    """Test that loader process with missing text throws and logs error"""
    caplog.set_level(logging.DEBUG)
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))

    # Mocks for the process wrapper method
    def dummy_fetch(self) -> None: # pylint: disable=redefined-builtin, unused-argument
        """Mock fetch method"""
        return None
    def dummy_convert(self) -> None: # pylint: disable=redefined-builtin, unused-argument
        """Mock convert method"""
        return None
    monkeypatch.setattr("RAGToolBox.loader.BaseLoader.fetch", dummy_fetch)
    monkeypatch.setattr("RAGToolBox.loader.BaseLoader.convert", dummy_convert)

    test_loader = BaseLoader(source = '', output_dir = '')
    test_loader.process()

    assert 'Warning: No text extracted from' in caplog.text
