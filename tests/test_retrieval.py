import os
import io
import pytest
import requests
import html2text
import pdfplumber
from Bio import Entrez

from retrieval import (
    BaseRetriever,
    NCBIRetriever,
    TextRetriever,
    HTMLRetriever,
    PDFRetriever,
    UnknownRetriever
    )

# Helpers and Mocks
class DummyPage:

    def __init__(self, text):
        self._text = text
    def extract_text(self):
        return self._text

class DummyPDF:

    def __init__(self, pages):
        self.pages = pages
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

class DummyResponse:

    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code
    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(f"Status code: {self.status_code}")

class DummyHandle:

    def __init__(self, text):
        self._text = text
    def read(self):
        return self._text
    def close(self):
        pass


# Unit tests
def test_detect_retriever_txt():
    cls = BaseRetriever.detect_retriever('http://example.com/file.txt', b'hello')
    assert cls is TextRetriever

def test_detect_retriever_md():
    cls = BaseRetriever.detect_retriever('http://example.com/file.md', b'')
    assert cls is TextRetriever

def test_detect_retriever_html():
    cls = BaseRetriever.detect_retriever('http://example.com/index.html', b'<html>')
    assert cls is HTMLRetriever

def test_detect_retriever_pdf():
    cls = BaseRetriever.detect_retriever('http://example.com/doc.pdf', b'%PDF')
    assert cls is PDFRetriever

def test_detect_retriever_unknown():
    cls = BaseRetriever.detect_retriever('http://example.com/archive.zip', b'PK')
    assert cls is UnknownRetriever

def test_detect_retriever_ncbi():
    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC123456/'
    cls = BaseRetriever.detect_retriever(url, b'ignored')
    assert cls is NCBIRetriever

def test_ncbi_retriever_fetch_success(monkeypatch):
    calls = []
    def dummy_efetch(db, id, rettype, retmode):
        calls.append((db, id, rettype, retmode))
        return DummyHandle('Abstract text')
    monkeypatch.setattr(Entrez, 'efetch', dummy_efetch)

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC999999/'
    retr = NCBIRetriever(url, 'out')
    retr.fetch()
    assert retr.raw_content == b'Abstract text'

    retr.convert()
    assert retr.text == 'Abstract text'
    assert calls[0] == ('pmc', 'PMC999999', 'abstract', 'text')

def test_ncbi_retriever_fetch_failure(monkeypatch):
    def dummy_efetch(db, id, rettype, retmode):
        raise Exception('NCBI down')
    monkeypatch.setattr(Entrez, 'efetch', dummy_efetch)

    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC000000/'
    retr = NCBIRetriever(url, 'out')
    with pytest.raises(RuntimeError) as exc:
        retr.fetch()
    assert 'Entrez fetch failed for PMC000000' in str(exc.value)

def test_text_retriever_convert():
    tr = TextRetriever('http://example.com/test.txt', 'out')
    tr.raw_content = 'Hello, World!'.encode('utf-8')
    tr.convert()
    assert tr.text == 'Hello, World!'

def test_html_retriever_convert():
    html = '<html><body><h1>Hi</h1><p>Para</p></body></html>'
    hr = HTMLRetriever('http://example.com/index.html', 'out')
    hr.raw_content = html.encode('utf-8')
    hr.convert()
    # Should contain markdown headings and paragraph text
    assert 'Hi' in hr.text
    assert 'Para' in hr.text

def test_pdf_retriever_convert(monkeypatch):
    # Mock pdfplumber.open to return DummyPDF with pages
    pages = [DummyPage('Page1'), DummyPage('Page2')]
    monkeypatch.setattr(pdfplumber, 'open', lambda _: DummyPDF(pages))
    pr = PDFRetriever('http://example.com/doc.pdf', 'out')
    pr.raw_content = b'%PDF-1.4 dummy'
    pr.convert()
    assert 'Page1' in pr.text
    assert 'Page2' in pr.text

def test_unknown_retriever_convert(capsys):
    ur = UnknownRetriever('http://example.com/file.xyz', 'out')
    ur.raw_content = b''
    ur.convert()
    captured = capsys.readouterr()
    assert 'Unknown format' in captured.out
    assert ur.text == ''

def test_save(tmp_path):
    tr = TextRetriever('http://example.com/one.txt', tmp_path)
    tr.text = 'TestSave'
    tr.save()
    out_file = tmp_path / 'one.txt'
    assert out_file.exists()
    assert out_file.read_text(encoding='utf-8') == 'TestSave'

# Integration tests
def test_full_process_text(monkeypatch, tmp_path):
    # Mock requests.get for a TXT URL
    monkeypatch.setattr(requests, 'get', lambda url: DummyResponse(b'Some text content'))
    url = 'http://example.com/data.txt'

    RetrieverClass = BaseRetriever.detect_retriever(url, b'some')
    retr = RetrieverClass(url, str(tmp_path))
    retr.process()

    out_file = tmp_path / 'data.txt'
    assert out_file.exists()
    assert 'Some text content' in out_file.read_text(encoding='utf-8')

def test_full_process_html(monkeypatch, tmp_path):
    html = '<html><body><p>Hello HTML</p></body></html>'
    monkeypatch.setattr(requests, 'get', lambda url: DummyResponse(html.encode('utf-8')))
    url = 'http://example.com/page.html'

    RetrieverClass = BaseRetriever.detect_retriever(url, html.encode('utf-8'))
    retr = RetrieverClass(url, str(tmp_path))
    retr.process()

    out_file = tmp_path / 'page.txt'
    assert out_file.exists()
    assert 'Hello HTML' in out_file.read_text(encoding='utf-8')

def test_full_process_pdf(monkeypatch, tmp_path):
    # Mock requests.get and pdfplumber.open
    monkeypatch.setattr(requests, 'get', lambda url: DummyResponse(b'%PDF dummy content'))
    pages = [DummyPage('X'), DummyPage('Y')]
    monkeypatch.setattr(pdfplumber, 'open', lambda _: DummyPDF(pages))

    url = 'http://example.com/report.pdf'
    RetrieverClass = BaseRetriever.detect_retriever(url, b'%PDF')
    retr = RetrieverClass(url, str(tmp_path))
    retr.process()

    out_file = tmp_path / 'report.txt'
    assert out_file.exists()
    text = out_file.read_text(encoding='utf-8')
    assert 'X' in text
    assert 'Y' in text

def test_full_process_ncbi(monkeypatch, tmp_path):
    # Monkeypatch efetch to return dummy abstract text
    class DummyHandle2:
        def __init__(self, text): self._text = text
        def read(self): return self._text
        def close(self): pass

    monkeypatch.setattr(Entrez, 'efetch', lambda db, id, rettype, retmode: DummyHandle2('Integration abstract text'))
    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC111111/'
    RetrieverClass = BaseRetriever.detect_retriever(url, b'ignored')
    retr = RetrieverClass(url, str(tmp_path))
    retr.process()
    # Check that file was created and contains the abstract
    out_file = tmp_path / 'PMC111111.txt'
    assert out_file.exists()
    content = out_file.read_text(encoding='utf-8')
    assert 'Integration abstract text' in content

def test_intentional_failure():
    assert False, "This failure is intentional to prove CI breaks on non-zero exit"
