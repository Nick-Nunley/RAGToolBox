import os
import pytest
import requests
import pdfplumber
from Bio import Entrez
from typing import Any

from src.loader import (
    BaseLoader,
    NCBILoader,
    TextLoader,
    HTMLLoader,
    PDFLoader,
    UnknownLoader
    )

# Helpers and Mocks
class DummyPage:
    def __init__(self, text: str) -> None:
        self._text = text
    def extract_text(self) -> str:
        return self._text

class DummyPDF:
    def __init__(self, pages: list) -> None:
        self.pages = pages
    def __enter__(self) -> 'DummyPDF':
        return self
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return False

class DummyResponse:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code
    def raise_for_status(self) -> None:
        if self.status_code != 200:
            raise requests.HTTPError(f"Status code: {self.status_code}")

class DummyHandle:
    def __init__(self, text: bytes) -> None:
        self._text = text
    def read(self) -> bytes:
        return self._text
    def close(self) -> None:
        pass


# Unit tests
def test_detect_loader_txt() -> None:
    cls = BaseLoader.detect_loader('http://example.com/file.txt', b'hello')
    assert cls is TextLoader

def test_detect_loader_md() -> None:
    cls = BaseLoader.detect_loader('http://example.com/file.md', b'')
    assert cls is TextLoader

def test_detect_loader_html() -> None:
    cls = BaseLoader.detect_loader('http://example.com/index.html', b'<html>')
    assert cls is HTMLLoader

def test_detect_loader_pdf() -> None:
    cls = BaseLoader.detect_loader('http://example.com/doc.pdf', b'%PDF')
    assert cls is PDFLoader

def test_detect_loader_unknown() -> None:
    cls = BaseLoader.detect_loader('http://example.com/archive.zip', b'PK')
    assert cls is UnknownLoader

def test_detect_loader_ncbi() -> None:
    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC123456/'
    cls = BaseLoader.detect_loader(url, b'ignored')
    assert cls is NCBILoader

def test_detect_loader_local_txt() -> None:
    # Mock file existence check
    original_exists = os.path.exists
    
    def mock_exists(path: str) -> bool:
        if path == '/path/to/file.txt':
            return True
        return original_exists(path)
    
    # Temporarily replace os.path.exists
    os.path.exists = mock_exists
    try:
        cls = BaseLoader.detect_loader('/path/to/file.txt', b'hello world')
        assert cls is TextLoader
    finally:
        os.path.exists = original_exists

def test_detect_loader_local_pdf() -> None:
    # Mock file existence check
    original_exists = os.path.exists
    
    def mock_exists(path: str) -> bool:
        if path == '/path/to/document.pdf':
            return True
        return original_exists(path)
    
    # Temporarily replace os.path.exists
    os.path.exists = mock_exists
    try:
        cls = BaseLoader.detect_loader('/path/to/document.pdf', b'%PDF-1.4')
        assert cls is PDFLoader
    finally:
        os.path.exists = original_exists

def test_detect_loader_local_html() -> None:
    # Mock file existence check
    original_exists = os.path.exists
    
    def mock_exists(path: str) -> bool:
        if path == '/path/to/page.html':
            return True
        return original_exists(path)
    
    # Temporarily replace os.path.exists
    os.path.exists = mock_exists
    try:
        cls = BaseLoader.detect_loader('/path/to/page.html', b'<html><body>content</body></html>')
        assert cls is HTMLLoader
    finally:
        os.path.exists = original_exists

def test_ncbi_loader_fetch_success(monkeypatch: Any) -> None:
    calls = []
    def dummy_efetch(db: str, id: str, rettype: str, retmode: str) -> DummyHandle:
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

def test_ncbi_loader_fetch_failure(monkeypatch: Any) -> None:
    def dummy_efetch(db: str, id: str, rettype: str, retmode: str) -> None:
        raise Exception('NCBI down')
    monkeypatch.setattr(Entrez, 'efetch', dummy_efetch)

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC000000/'
    loader = NCBILoader(url, 'out')
    with pytest.raises(RuntimeError) as exc:
        loader.fetch()
    assert 'Entrez fetch failed for PMC000000' in str(exc.value)

def test_text_loader_convert() -> None:
    tr = TextLoader('http://example.com/test.txt', 'out')
    tr.raw_content = 'Hello, World!'.encode('utf-8')
    tr.convert()
    assert tr.text == 'Hello, World!'

def test_html_loader_convert() -> None:
    html = '<html><body><h1>Hi</h1><p>Para</p></body></html>'
    hr = HTMLLoader('http://example.com/index.html', 'out')
    hr.raw_content = html.encode('utf-8')
    hr.convert()
    # Should contain markdown headings and paragraph text
    assert 'Hi' in hr.text
    assert 'Para' in hr.text

def test_html_loader_navigablestring_markdown() -> None:
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

def test_pdf_loader_convert(monkeypatch: Any) -> None:
    # Mock pdfplumber.open to return DummyPDF with pages
    pages = [DummyPage('Page1'), DummyPage('Page2')]
    monkeypatch.setattr(pdfplumber, 'open', lambda _: DummyPDF(pages))
    pr = PDFLoader('http://example.com/doc.pdf', 'out')
    pr.raw_content = b'%PDF-1.4 dummy'
    pr.convert()
    assert 'Page1' in pr.text
    assert 'Page2' in pr.text

def test_unknown_loader_convert(capsys: Any) -> None:
    ur = UnknownLoader('http://example.com/file.xyz', 'out')
    ur.raw_content = b''
    ur.convert()
    captured = capsys.readouterr()
    assert 'Unknown format' in captured.out
    assert ur.text == ''

def test_save(tmp_path: Any) -> None:
    tr = TextLoader('http://example.com/one.txt', tmp_path)
    tr.text = 'TestSave'
    tr.save()
    out_file = tmp_path / 'one.txt'
    assert out_file.exists()
    assert out_file.read_text(encoding='utf-8') == 'TestSave'

# Integration tests
def test_full_process_text(monkeypatch: Any, tmp_path: Any) -> None:
    # Mock requests.get for a TXT URL
    monkeypatch.setattr(requests, 'get', lambda url: DummyResponse(b'Some text content'))
    url = 'http://example.com/data.txt'

    LoaderClass = BaseLoader.detect_loader(url, b'some')
    loader = LoaderClass(url, str(tmp_path))
    loader.process()

    out_file = tmp_path / 'data.txt'
    assert out_file.exists()
    assert 'Some text content' in out_file.read_text(encoding='utf-8')

def test_full_process_html(monkeypatch: Any, tmp_path: Any) -> None:
    html = '<html><body><p>Hello HTML</p></body></html>'
    monkeypatch.setattr(requests, 'get', lambda url: DummyResponse(html.encode('utf-8')))
    url = 'http://example.com/page.html'

    LoaderClass = BaseLoader.detect_loader(url, html.encode('utf-8'))
    loader = LoaderClass(url, str(tmp_path))
    loader.process()

    out_file = tmp_path / 'page.txt'
    assert out_file.exists()
    assert 'Hello HTML' in out_file.read_text(encoding='utf-8')

def test_full_process_pdf(monkeypatch: Any, tmp_path: Any) -> None:
    # Mock requests.get and pdfplumber.open
    monkeypatch.setattr(requests, 'get', lambda url: DummyResponse(b'%PDF dummy content'))
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
    # Monkeypatch efetch to return dummy PMC XML with an abstract
    class DummyHandle2:
        def __init__(self, text: bytes) -> None: self._text = text
        def read(self) -> bytes: return self._text
        def close(self) -> None: pass

    xml = b'<?xml version="1.0"?><article><abstract>Integration abstract text</abstract></article>'
    monkeypatch.setattr(Entrez, 'efetch', lambda db, id, rettype, retmode: DummyHandle2(xml))
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
    # Monkeypatch efetch to return dummy PubMed XML with an abstract and title
    class DummyHandle2:
        def __init__(self, text: bytes) -> None: self._text = text
        def read(self) -> bytes: return self._text
        def close(self) -> None: pass

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
    monkeypatch.setattr(Entrez, 'efetch', lambda db, id, rettype, retmode: DummyHandle2(xml))
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
    # Create a temporary text file
    test_file = tmp_path / 'test_document.txt'
    test_content = 'This is a test document with some content.'
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    # Process the local file
    loader = TextLoader(str(test_file), str(tmp_path / 'output'))
    loader.process()
    
    # Check that output file was created
    output_file = tmp_path / 'output' / 'test_document.txt'
    assert output_file.exists()
    
    # Check content
    with open(output_file) as f:
        content = f.read()
        assert content == test_content

def test_full_process_local_pdf(monkeypatch: Any, tmp_path: Any) -> None:
    # Mock pdfplumber.open to return DummyPDF with pages
    def mock_pdf_open(file_obj: Any) -> DummyPDF:
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
    with open(output_file) as f:
        content = f.read()
        assert 'Page 1 content' in content
        assert 'Page 2 content' in content
