"""
RAGToolBox Loader module.

Ingests documents from multiple sources (PubMed/PMC via NCBI, generic HTML,
PDF, and plain text) and converts them to normalized Markdown/TXT ready for
chunking and indexing.

Environment:
    NCBI_EMAIL: Email address used by NCBI E-utilities (recommended).
    (Optional) HTTP(S) proxy environment variables respected by requests.

CLI:
    This module can be invoked as a script to fetch one or more sources and
    save converted `.txt` files to an output directory.

Additionally, this script provides a CLI entry point for execution as a standalone python module.
"""

import argparse
import os
import logging
import io
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from typing import Optional, Dict, Any, List, Union, Tuple
import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString
import html2text
import pdfplumber
from Bio import Entrez
from RAGToolBox.logging import RAGTBLogger

__all__ = ['BaseLoader', 'NCBILoader', 'HTMLLoader', 'PDFLoader', 'TextLoader', 'UnknownLoader']
logger = logging.getLogger(__name__)

class BaseLoader:
    """
    Base class for fetching, converting, and saving content from a URL or local file.

    Subclasses implement :meth:`convert` to turn raw bytes into normalized text.
    Typical usage is :meth:`process`, which runs ``fetch -> convert -> save``.

    Attributes:
        source: Original input (URL or local file path)
        output_dir: Directory where converted text will be written
        raw_content: Raw bytes (or str for some XML cases) obtained by :meth:`fetch`
        text: Converted plain text/markdown produced by :meth:`convert`
        is_local_file: Whether ``source`` points to an on-disk file
        timeout: Per-request timeout (seconds) for remote fetches
    """
    source: str  # Can be URL or file path
    output_dir: str
    raw_content: Optional[Union[bytes, str]]
    text: str
    is_local_file: bool

    def __init__(self, source: str, output_dir: str, *, timeout: float = 30.0) -> None:
        """
        Initialize loader with a source and output directory.

        Args:
            source: URL or local file path
            output_dir: Directory to write the converted `.txt` file
            timeout: Timeout (seconds) for HTTP requests made during :meth:`fetch`
        """
        self.source = source
        self.output_dir = output_dir
        self.raw_content = None
        self.text = ""
        self.is_local_file = os.path.exists(source) and os.path.isfile(source)
        self.timeout = timeout

    def fetch(self) -> None:
        """
        Fetch raw content from `source`.

        If ``source`` is a local file, it is read as bytes. Otherwise, an HTTP
        GET is performed with ``timeout``.

        Raises:
            TimeoutError: If the HTTP request times out
            RuntimeError: For non-timeout HTTP errors (4xx/5xx) or connection failures
        """
        if self.is_local_file:
            with open(self.source, 'rb') as f:
                self.raw_content = f.read()
        else:
            try:
                logger.debug(self.timeout)
                response = requests.get(self.source, timeout=self.timeout)
                response.raise_for_status()
                self.raw_content = response.content
            except requests.Timeout as exc:
                err = f"Timed out after {self.timeout}s fetching {self.source}"
                logger.error(err, exc_info=True)
                raise TimeoutError(err) from exc
            except requests.RequestException as exc:
                # covers HTTP errors, connection errors, etc.
                err = f"Error fetching {self.source!r}"
                logger.error(err, exc_info=True)
                raise RuntimeError(err) from exc

    @staticmethod
    def _handle_local_file_detection(source: str, content: bytes) -> type:
        """Helper method for detect loader to handle local files"""
        ext = os.path.splitext(source)[1].lower()
        if ext == '.pdf':
            logger.debug('PDF detected while loading %s', source)
            return PDFLoader
        if ext in ['.txt', '.md']:
            logger.debug('TXT file detected while loading %s', source)
            return TextLoader
        if ext in ['.html', '.htm']:
            logger.debug('HTML detected while loading %s', source)
            return HTMLLoader
        # Try to detect by content
        head = content[:512].lstrip().lower()
        if head.startswith(b'<html') or head.startswith(b'<!doctype html'):
            logger.debug('HTML detected while loading %s', source)
            return HTMLLoader
        if head.startswith(b'%pdf'):
            logger.debug('PDF detected while loading %s', source)
            return PDFLoader
        logger.warning(
            'Unknown format detected while loading %s. Falling back to TextLoader.',
            source
            )
        return TextLoader

    @staticmethod
    def _handle_remote_file_detection(source: str, content: bytes) -> type:
        """Helper method for detect loader to handle remote files"""
        # URL-based detection (existing logic)
        path = urlparse(source).path.lower()
        ext = os.path.splitext(path)[1]
        head = content[:512].lstrip().lower()

        if 'ncbi.nlm.nih.gov' in urlparse(source).netloc:
            logger.debug('NCBI format detected while loading %s', source)
            return NCBILoader
        if ext in ['.html', '.htm'] or \
        head.startswith(b'<html') or \
        head.startswith(b'<!doctype html'):
            logger.debug('HTML detected while loading %s', source)
            return HTMLLoader
        if ext == '.pdf':
            logger.debug('PDF detected while loading %s', source)
            return PDFLoader
        if ext in ['.txt', '.md']:
            logger.debug('TXT file detected while loading %s', source)
            return TextLoader
        logger.warning(
            'Unknown format detected while loading %s. Falling back to UnknownLoader.',
            source
            )
        return UnknownLoader

    @staticmethod
    def detect_loader(source: str, content: bytes) -> type:
        """
        Choose an appropriate loader subclass for ``source`` and ``content``.

        Detection uses file extension, URL host (e.g., NCBI), and lightweight
        content sniffing (e.g., HTML/PDF signatures).

        Args:
            source: URL or local file path
            content: Initial bytes read/fetched from ``source`` (for sniffing)

        Returns:
            A :class:`BaseLoader` subclass such as :class:`HTMLLoader`,
            :class:`PDFLoader`, :class:`TextLoader`, :class:`NCBILoader`,
            or :class:`UnknownLoader`
        """
        # Check if source is a local file
        if os.path.exists(source) and os.path.isfile(source):
            return BaseLoader._handle_local_file_detection(source=source, content=content)
        return BaseLoader._handle_remote_file_detection(source=source, content=content)

    def convert(self) -> None:
        """
        Convert :attr:`raw_content` into normalized text.

        Subclasses must implement this method and set :attr:`text`.

        Raises:
            NotImplementedError: Always in the base class
        """
        err = "Subclasses must implement convert()"
        logger.error(err)
        raise NotImplementedError(err)

    def save(self) -> None:
        """
        Write :attr:`text` to ``output_dir`` as ``<name>.txt``.

        The filename is derived from:
            * local files: the original basename without extension, or
            * URLs: the path basename, falling back to ``document.txt``.

        Side Effects:
            Creates ``output_dir`` if missing and writes the file to disk
        """
        os.makedirs(self.output_dir, exist_ok=True)

        if self.is_local_file:
            # For local files, use the original filename without extension
            name = os.path.splitext(os.path.basename(self.source))[0]
        else:
            # For URLs, use the existing logic
            parsed = urlparse(self.source)
            name = os.path.splitext(os.path.basename(parsed.path) or 'document')[0]

        filename = f"{name}.txt"
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(self.text)
        logger.info("Saved plain text to %s", out_path)

    def process(self) -> None:
        """
        End-to-end pipeline: :meth:`fetch`, :meth:`convert`, and then :meth:`save`.

        Notes:
            If :attr:`text` is empty after conversion, the file is not saved and a
            warning is logged
        """
        logger.info("Processing: %s", self.source)
        self.fetch()
        self.convert()
        if self.text:
            self.save()
        else:
            logger.warning("Warning: No text extracted from %s", self.source)

class NCBILoader(BaseLoader):
    """
    Loader for PubMed and PMC articles via NCBI E-utilities.

    Fetches XML (and, for PMC, optionally follows to the PDF), extracts metadata,
    abstract, body (if available), and references, and renders Markdown-like text.

    Attributes:
        pmc_id: Identifier parsed from the URL path (PMC or PubMed)
        article_data: Parsed metadata and sections (title, authors, journal, etc.)
        _used_pdf: Whether a PMC PDF was downloaded and used as the primary source
    """

    pmc_id: str
    _used_pdf: bool
    article_data: Optional[Dict[str, Any]]

    def __init__(self, source: str, output_dir: str) -> None:
        """
        Initializes an instance of NCBILoader.

        Attributes:
            pmc_id: Identifier parsed from the URL path (PMC or PubMed)
            article_data: Parsed metadata and sections (title, authors, journal, etc.)
            _used_pdf: Whether a PMC PDF was downloaded and used as the primary source
            _supported_sources: List of strings to validate ``source_type`` args against
        """
        super().__init__(source, output_dir)
        self.pmc_id = os.path.basename(urlparse(self.source).path.rstrip('/'))
        self._used_pdf = False
        self.article_data = None
        self._supported_sources = ['PMC', 'PubMed']

    def _get_ncbi_db_from_url(self) -> str:
        """Helper method for determining which NCBI source to use"""
        netloc = urlparse(self.source).netloc
        if 'pmc.' in netloc:
            return 'pmc'
        if 'pubmed.' in netloc:
            return 'pubmed'
        # Fallback: assume PMC
        logger.warning(
            "Detected NCBI format, but could not find 'pmc.' or 'pubmed.' strings in %s",
            netloc
            )
        return 'pmc'

    def fetch(self) -> None:
        """
        Fetch XML with Entrez and prefer PMC PDF when available.


        Uses :func:`Bio.Entrez.efetch` to retrieve article XML. For PMC records,
        attempts to locate and download a PDF and extract text via :class:`PDFLoader`.

        Raises:
            RuntimeError: If the Entrez fetch fails
        """
        pmc_id = os.path.basename(urlparse(self.source).path.rstrip('/'))
        db = self._get_ncbi_db_from_url()
        tried: List[str] = []
        # Fetch XML from the correct db
        try:
            handle = Entrez.efetch(
                db=db,
                id=pmc_id,
                rettype="full" if db == "pmc" else "xml",
                retmode="xml"
                )
            xml_content = handle.read()
            handle.close()
            self.raw_content = xml_content
        except Exception as e:
            tried.append(f"{db}/xml: {e}")
            err = f"Entrez fetch failed for {pmc_id}: {tried}"
            logger.error(err, exc_info=True)
            raise RuntimeError(err) from e

        # Try to extract PDF link if PMC
        pdf_url = self._extract_pdf_url_from_xml(
            self.raw_content
            ) if db == 'pmc' and self.raw_content else None
        if pdf_url:
            logger.debug(
                "PDF link found for %s: %s\nAttempting to download and extract text from PDF.",
                pmc_id,
                pdf_url
                )
            try:
                pdf_bytes = self._download_pdf(pdf_url)
                pdf_loader = PDFLoader(self.source, self.output_dir)
                pdf_loader.raw_content = pdf_bytes
                pdf_loader.convert()
                self.text = pdf_loader.text
                self._used_pdf = True
                return
            except Exception as _: # pylint: disable=broad-exception-caught
                logger.warning(
                    "Failed to download or process PDF for %s\nFalling back to XML extraction.",
                    pmc_id,
                    exc_info=True
                    )
                self._used_pdf = False
        else:
            self._used_pdf = False

        # If no PDF, check for 'not allowed' comment and warn (handle both bytes and str)
        if self.raw_content is None:
            return
        raw_str: str = self.raw_content.decode(
            'utf-8',
            errors='ignore'
            ) if isinstance(self.raw_content, bytes) else self.raw_content
        if "does not allow downloading of the full text" in raw_str:
            logger.warning(
                "Warning: Full text not available for %s. Only abstract and metadata "
                "will be extracted.",
                pmc_id
                )

    def _extract_pdf_url_from_xml(self, xml_bytes: Union[bytes, str]) -> Optional[str]:
        """Parse XML and extract the PDF link if present."""
        try:
            root = ET.fromstring(xml_bytes)
            for self_uri in root.findall('.//self-uri'):
                if self_uri.attrib.get('content-type') == 'pmc-pdf':
                    href = self_uri.attrib.get('{http://www.w3.org/1999/xlink}href')
                    if href:
                        if href.startswith('http'):
                            return href
                        pmc_id = os.path.basename(urlparse(self.source).path.rstrip('/'))
                        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/{href}"
            return None
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"Error parsing XML for PDF link: {e}")
            logger.warning("Error parsing XML for PDF link. Returning dtype=None.", exc_info=True)
            return None

    def _download_pdf(self, pdf_url: str) -> bytes:
        response = requests.get(pdf_url, timeout=self.timeout)
        response.raise_for_status()
        return response.content

    def _safe_text(self, elem: Optional[ET.Element]) -> str:
        """Text processing method to strip out strings from XML elements"""
        return elem.text.strip() if elem is not None and elem.text else ""

    def _extract_all_text(self, elem: Optional[ET.Element]) -> str:
        """Method to extract text out of XML content"""
        if elem is None:
            return ""
        texts: List[str] = []
        if elem.text:
            texts.append(elem.text.strip())
        for child in elem:
            texts.append(self._extract_all_text(child))
        if elem.tail:
            texts.append(elem.tail.strip())
        return " ".join([t for t in texts if t])

    def _convert_helper(self) -> None:
        text_parts: List[str] = []
        if self.article_data.get('title'):
            text_parts.append(f"# {self.article_data['title']}\n")
        metadata_parts: List[str] = []
        if self.article_data.get('authors'):
            authors_text = ", ".join(self.article_data['authors'])
            metadata_parts.append(f"**Authors:** {authors_text}")
        if self.article_data.get('journal'):
            metadata_parts.append(f"**Journal:** {self.article_data['journal']}")
        if self.article_data.get('doi'):
            metadata_parts.append(f"**DOI:** {self.article_data['doi']}")
        if self.article_data.get('keywords'):
            keywords_text = ", ".join(self.article_data['keywords'])
            metadata_parts.append(f"**Keywords:** {keywords_text}")
        if metadata_parts:
            text_parts.append("\n".join(metadata_parts))
            text_parts.append("\n")
        text_parts.append("---\n")
        if self.article_data.get('abstract'):
            clean_abstract = self._clean_text(self.article_data['abstract'])
            if '<' in clean_abstract and '>' in clean_abstract:
                clean_abstract = html2text.html2text(clean_abstract)
            text_parts.append("## Abstract\n")
            text_parts.append(clean_abstract)
            text_parts.append("\n")
        if self.article_data.get('body'):
            clean_body = self._clean_text(self.article_data['body'])
            if '<' in clean_body and '>' in clean_body:
                clean_body = html2text.html2text(clean_body)
            text_parts.append("## Main Text\n")
            text_parts.append(clean_body)
            text_parts.append("\n")
        if self.article_data.get('references'):
            clean_refs = self._clean_text(self.article_data['references'])
            if '<' in clean_refs and '>' in clean_refs:
                clean_refs = html2text.html2text(clean_refs)
            text_parts.append("## References\n")
            text_parts.append(clean_refs)
        self.text = "".join(text_parts)

    def convert(self) -> None:
        """
        Convert fetched XML (or PDF result) into Markdown-like text.

        If a PMC PDF was already processed during :meth:`fetch`, this is a no-op.
        Otherwise, parses the XML into :attr:`article_data` and formats sections.

        Side Effects:
            Sets :attr:`text` to the final normalized content (or empty string if
            parsing fails)
        """
        if getattr(self, '_used_pdf', False) and self.text:
            return
        self.article_data = self._parse_xml_content()
        if not self.article_data:
            self.text = ""
            return
        self._convert_helper()

    def _parse_xml_content(self) -> Optional[Dict[str, Any]]:
        """Dynamically parse PMC or PubMed XML content."""
        if self.raw_content is None:
            return None
        try:
            root = ET.fromstring(self.raw_content)
            tag = root.tag.lower()
            # Detect PMC or PubMed XML
            if tag.endswith('pmc-articleset') or tag.endswith('article'):
                return self._parse_pmc_xml(root)
            if tag.endswith('pubmedarticleset') or tag.endswith('pubmedarticle'):
                return self._parse_pubmed_xml(root)
            logger.warning("Unkonwn XML root tag: %s. Returning empty article data.", tag)
            return {}
        except Exception as _: # pylint: disable=broad-exception-caught
            logger.warning("Error parsing XML content. Returning dtype=None.", exc_info=True)
            return None

    def _check_available_sources(self, source_type: str) -> None:
        if source_type not in self._supported_sources:
            err = (
                f'{source_type} is not supported by NCBILoader. '
                f'See available sources: {self._supported_sources}'
                )
            logger.error(err)
            raise ValueError(err)

    def _obtain_authors(self, root: ET.Element, source_type: str) -> List[str]:
        self._check_available_sources(source_type=source_type)
        authors_list: List[str] = []
        if source_type == 'PMC':
            for author_elem in root.findall(".//contrib[@contrib-type='author']"):
                surname = author_elem.find(".//surname")
                given_names = author_elem.find(".//given-names")
                if surname is not None and given_names is not None:
                    authors_list.append(
                        f"{self._safe_text(given_names)} {self._safe_text(surname)}"
                        )
            return authors_list
        for author_elem in root.findall(".//AuthorList/Author"):
            last = self._safe_text(author_elem.find("LastName"))
            fore = self._safe_text(author_elem.find("ForeName"))
            if last or fore:
                authors_list.append(f"{fore} {last}".strip())
        return authors_list

    def _obtain_keywords(self, root: ET.Element, source_type: str) -> List[str]:
        self._check_available_sources(source_type=source_type)
        keywords_list: List[str] = []
        if source_type == 'PMC':
            for keyword_elem in root.findall(".//kwd"):
                kw = self._safe_text(keyword_elem)
                if kw:
                    keywords_list.append(kw)
            return keywords_list
        for mesh_elem in root.findall(".//MeshHeading/DescriptorName"):
            kw = self._safe_text(mesh_elem)
            if kw:
                keywords_list.append(kw)
        return keywords_list

    def _obtain_references(self, root: ET.Element, source_type: str) -> List[str]:
        self._check_available_sources(source_type=source_type)
        references_list: List[str] = []
        if source_type == 'PMC':
            for ref_elem in root.findall(".//ref"):
                ref_text = self._extract_all_text(ref_elem)
                if ref_text:
                    references_list.append(ref_text)
            return references_list
        for ref_elem in root.findall(".//ReferenceList/Reference/Citation"):
            ref_text = self._extract_all_text(ref_elem)
            if ref_text:
                references_list.append(ref_text)
        return references_list

    def _parse_pmc_xml(self, root: ET.Element) -> Dict[str, Any]:
        article_data: Dict[str, Any] = {}
        # Title
        title_elem = root.find(".//article-title")
        article_data['title'] = self._safe_text(title_elem)
        # Authors
        article_data['authors'] = self._obtain_authors(root=root, source_type='PMC')
        # Journal
        journal_elem = root.find(".//journal-title")
        article_data['journal'] = self._safe_text(journal_elem)
        # DOI
        doi_elem = root.find(".//article-id[@pub-id-type='doi']")
        article_data['doi'] = self._safe_text(doi_elem)
        # Keywords
        article_data['keywords'] = self._obtain_keywords(root=root, source_type='PMC')
        # Abstract
        abstract_elem = root.find(".//abstract")
        article_data['abstract'] = self._extract_all_text(abstract_elem)
        # Body
        body_elem = root.find(".//body")
        article_data['body'] = self._extract_all_text(body_elem)
        if not article_data['body']:
            logger.warning(
                "Warning: Full text/body not availble for %s. "
                "Only abstract and metadata will be extracted.",
                self.pmc_id
                )
        # References
        article_data['references'] = "\n".join(
            self._obtain_keywords(root=root, source_type='PMC')
            )
        return article_data

    def _parse_pubmed_xml(self, root: ET.Element) -> Dict[str, Any]:
        article_data: Dict[str, Any] = {}
        # Title
        title_elem = root.find(".//ArticleTitle")
        article_data['title'] = self._safe_text(title_elem)
        # Authors
        article_data['authors'] = self._obtain_authors(root=root, source_type='PubMed')
        # Journal
        journal_elem = root.find(".//Journal/Title")
        article_data['journal'] = self._safe_text(journal_elem)
        # DOI
        doi_elem = root.find(".//ELocationID[@EIdType='doi']")
        article_data['doi'] = self._safe_text(doi_elem)
        # Keywords (PubMed Mesh terms)
        article_data['keywords'] = self._obtain_keywords(root=root, source_type='PubMed')
        # Abstract
        abstract_elem = root.find(".//Abstract")
        article_data['abstract'] = self._extract_all_text(abstract_elem)
        # Body (not available in PubMed, leave blank)
        article_data['body'] = ""
        # References
        article_data['references'] = "\n".join(
            self._obtain_references(root=root, source_type='PubMed')
            )
        return article_data

    def _clean_text(self, text: str) -> str:
        """Clean up text by removing common HTML tags and extra whitespace."""
        text = re.sub(r'<[^>]+>', '', text) # Remove all HTML tags
        text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with single space
        text = text.strip()
        return text

    def save(self) -> None:
        """
        Save converted text using the PMC/PubMed identifier as the filename.

        Side Effects:
            Writes ``<pmc_id>.txt`` into ``output_dir``
        """
        os.makedirs(self.output_dir, exist_ok=True)
        filename = f"{self.pmc_id}.txt"
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(self.text)
        logger.info("Saved plain text to %s", out_path)

class HTMLLoader(BaseLoader):
    """
    Loader for generic HTML pages.

    Extracts a reasonable article title and body (using Readability as needed),
    converts headings, paragraphs, lists, and blockquotes to Markdown-like text,
    and saves the result using a slugified title when available.
    """

    def __init__(self, source, output_dir, use_readability: bool = False):
        """
        Initializes an instance of HTMLLoader.

        Args:
            source: Page URL
            output_dir: Directory to write output
            use_readability: If ``True``, favor Readability extraction when the
                initial conversion is too short/low-signal
        """
        super().__init__(source, output_dir)
        self._use_readability = use_readability

    def _slugify(self, text: str, maxlen: int = 80) -> str:
        text = text.strip().lower()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        return text[:maxlen].rstrip('-')

    def _find_main_container(self, soup: BeautifulSoup) -> BeautifulSoup:
        # Known “content” containers
        for sel in (
            "#mw-content-text",
            "div#content",
            "div.main-content",
            "div.article-content",
            "div#__nuxt",
            "div#root",
            ):
            el = soup.select_one(sel)
            if el:
                return el  # type: ignore
        # HTML5 semantic
        for tag in ("article", "main"):
            el = soup.find(tag)
            if el and hasattr(el, 'find'):
                try:
                    if el.find("p"):  # type: ignore
                        return el  # type: ignore
                except AttributeError:
                    continue
        # Fallback on the whole soup
        return soup

    def _clean_lines(self, text_block: str) -> str:
        return "\n".join(line.strip() for line in text_block.splitlines() if line.strip())

    def _convert_helper(self, html: bytes) -> Tuple[str, str]:
        html_str = html.decode("utf-8", errors="ignore") if isinstance(html, bytes) else html

        # Use readability-lxml to extract the main article and title
        from readability import Document
        doc = Document(html_str)
        # Always call _extract_title with bytes
        title = doc.short_title() or self._extract_title(
            html if isinstance(html, bytes) else html.encode("utf-8", errors="ignore")
            )
        summary_html = doc.summary()
        return title, summary_html

    def _html_to_markdown_handle_headers(self, element) -> str:
        if isinstance(element, NavigableString):
            return str(element)
        if element.name == "h1":
            str_val = f"# {element.get_text(separator=' ', strip=True)}\n"
        elif element.name == "h2":
            str_val = f"## {element.get_text(separator=' ', strip=True)}\n"
        elif element.name == "h3":
            str_val = f"### {element.get_text(separator=' ', strip=True)}\n"
        elif element.name == "h4":
            str_val = f"#### {element.get_text(separator=' ', strip=True)}\n"
        elif element.name == "h5":
            str_val = f"##### {element.get_text(separator=' ', strip=True)}\n"
        elif element.name == "h6":
            str_val = f"###### {element.get_text(separator=' ', strip=True)}\n"
        elif element.name == "p":
            str_val = element.get_text(separator=' ', strip=True) + "\n"
        else:
            logger.warning(
                "Warning: no HTML header/section tags detected when converting HTML header "
                "to markdown. Returning dtype=None."
                )
            return None
        return str_val

    def _html_to_markdown(self, element) -> str:
        element_name = self._html_to_markdown_handle_headers(element=element)
        if element_name is not None:
            return element_name
        if element.name == "ul":
            return "\n".join(
                f"- {li.get_text(separator=' ', strip=True)}" for li in element.find_all(
                    "li", recursive=False
                    )
                ) + "\n"
        if element.name == "ol":
            return "\n".join(
                f"1. {li.get_text(separator=' ', strip=True)}" for li in element.find_all(
                    "li", recursive=False
                    )
                ) + "\n"
        if element.name == "blockquote":
            return "> " + element.get_text(separator=' ', strip=True) + "\n"
            # Recursively process children
        return "".join(self._html_to_markdown(child) for child in element.children)

    def convert(self) -> None:
        """
        Extract main content and render Markdown-like text.

        Behavior:
            * Use Readability to detect article title and body
            * Convert common HTML structures to Markdown headings/lists/quotes
            * Fallback to a simpler text-cleaning pipeline if content is too short

        Side Effects:
            Sets :attr:`text` to the normalized content (or empty string on failure)
        """
        if self.raw_content is None:
            logger.warning(
                "Warning: no content detected when calling HTMLLoader.convert(). "
                "Setting HTMLLoader.text attribute to an empty string."
                )
            self.text = ""
            return

        title, summary_html = self._convert_helper(self.raw_content)
        soup = BeautifulSoup(summary_html, "html.parser")

        # Build markdown content
        md_lines = []

        # Metadata block
        md_lines.append(f"# {title}\n")
        md_lines.append(f"**Source:** {self.source}")
        md_lines.append("---\n")

        # Convert HTML structure to markdown

        # Process the main content
        if soup.body:
            children = soup.body.children
        else:
            children = soup.children
        for child in children:
            if hasattr(child, "name") or isinstance(child, NavigableString):
                md = self._html_to_markdown(child)
                if md.strip():
                    md_lines.append(md)

        # Fallback: if markdown is too short, use previous logic
        content = "\n".join(md_lines).strip()
        if len(content) < 200:
            main = self._find_main_container(BeautifulSoup(self.raw_content, "html.parser"))
            raw_content = main.get_text("\n", strip=True)
            raw_content = re.sub(
                r"\[\s*edit(?: on [^\]]*)?\s*\]",
                "",
                raw_content,
                flags=re.IGNORECASE
                )
            body, refs = raw_content.split(
                "\nReferences",
                1
                ) if "\nReferences" in raw_content else (raw_content, "")
            clean = self._clean_lines(body)
            if refs:
                clean += "\n\nReferences\n" + self._clean_lines(refs)
            content = html2text.html2text(clean)

        self.text = content

    def _extract_title(self, html: bytes) -> str:
        soup = BeautifulSoup(html, "html.parser")
        # 1) try citation_title
        meta = soup.find("meta", {"name": "citation_title"})
        if meta and hasattr(meta, 'get') and meta.get("content"):  # type: ignore
            return meta["content"]  # type: ignore
        # 2) try OpenGraph
        og = soup.find("meta", {"property": "og:title"})
        if og and hasattr(og, 'get') and og.get("content"):  # type: ignore
            return og["content"]  # type: ignore
        # 3) fallback to <title>
        if soup.title and soup.title.string:
            return soup.title.string
        return ""

    def save(self) -> None:
        """
        Write the converted text using a slugified title when available.

        If no suitable title is found, falls back to the URL path basename.

        Side Effects:
            Writes ``<title-slug>.txt`` (or fallback) into ``output_dir``
        """
        os.makedirs(self.output_dir, exist_ok=True)
        # Try extracting a real title:
        if self.raw_content is None:
            logger.warning(
                "Warning: no content detected when calling HTMLLoader.save(). "
                "Setting title var to an empty string."
                )
            title = ""
        else:
            title = self._extract_title(self.raw_content)  # type: ignore
        if title:
            name = self._slugify(title)
        else:
            # Fallback to base class logic
            logger.warning("Warning: No title detected in HTML content for %s", self.source)
            parsed = urlparse(self.source)
            basename = os.path.basename(parsed.path) or "document"
            name = os.path.splitext(basename)[0]
        filename = f"{name}.txt"
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(self.text)
        logger.info("Saved plain text to %s", out_path)

class PDFLoader(BaseLoader):
    """
    Loader for PDFs.

    Uses :mod:`pdfplumber` to extract page text and concatenates the result
    into a plain text/Markdown-friendly form
    """

    def convert(self) -> None:
        """
        Extract text from each page of a PDF.

        Raises:
            RuntimeError: If PDF parsing fails (propagated from `pdfplumber`)
        """
        text_chunks = []
        if self.raw_content is None:
            logger.warning(
                "Warning: no content detected when calling PDFLoader.convert(). "
                "Setting PDFLoader.text attribute to an empty string."
                )
            return
        with pdfplumber.open(io.BytesIO(self.raw_content)) as pdf:  # type: ignore
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    text_chunks.append(txt)
        self.text = "\n".join(text_chunks)

class TextLoader(BaseLoader):
    """
    Loader for plain text or Markdown-like files.

    Decodes raw bytes as UTF-8 (ignoring errors) to produce the final text.
    """

    def convert(self) -> None:
        """DDecode :attr:`raw_content` as UTF-8 text (errors ignored)."""
        if self.raw_content is None:
            logger.warning(
                "Warning: no content detected when calling TextLoader.convert(). "
                "Setting TextLoader.text attribute to an empty string."
                )
            self.text = ""
        else:
            self.text = self.raw_content.decode('utf-8', errors = 'ignore')  # type: ignore

class UnknownLoader(BaseLoader):
    """
    Fallback loader for unrecognized formats.

    Leaves :attr:`text` empty and logs a warning. Useful to keep batch jobs
    moving when a source type is unsupported.
    """

    def convert(self) -> None:
        """Skip conversion for unknown formats (no-op, logs a warning)."""
        logger.warning("Unknown format for URL: %s. Cannot convert to text format.", self.source)
        self.text = ''


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Fetch multiple URLs or local files and " + \
        "convert content to plain text for chunking"
    )
    parser.add_argument(
        'sources', nargs='+', help='One or more URLs or local file paths to ingest'
    )
    parser.add_argument(
        '--output-dir', '-o', default='assets/kb',
        help='Directory where converted text files will be saved'
    )
    parser.add_argument(
        '--email', '-e', help='Email address for NCBI E-utilities',
        default=os.getenv('NCBI_EMAIL')
    )
    parser.add_argument(
        '--use-readability',
        action='store_true',
        help='If set, HTMLRetriever will fall back to Readability when the extracted text is short'
    )
    parser.add_argument(
        '--timeout',
        default=30.0,
        type=float,
        help='Time to wait when making requests to prevent hanging programs'
    )
    RAGTBLogger.add_logging_args(parser=parser)
    args = parser.parse_args()

    RAGTBLogger.configure_logging_from_args(args=args)

    # Set Entrez email if provided
    if args.email:
        Entrez.email = args.email
    else:
        if not Entrez.email:
            logger.warning(
                "Warning: No email provided for NCBI E-utilities; they may block requests."
                )

    for raw_source in args.sources:
        try:
            # Check if source is a local file
            if os.path.exists(raw_source) and os.path.isfile(raw_source):
                # For local files, read directly
                with open(raw_source, 'rb') as source_file:
                    raw = source_file.read()
            else:
                # For URLs, fetch via requests
                raw = requests.get(raw_source, timeout=args.timeout).content
        except TimeoutError as e:
            ERR = f"Request to {raw_source} timed out"
            logger.error(ERR, exc_info=True)
            raise TimeoutError(ERR) from e
        except Exception as e: # pylint: disable=broad-exception-caught
            logger.warning("Failed to fetch %s.", raw_source)
            continue

        LoaderClass = BaseLoader.detect_loader(raw_source, raw)
        if LoaderClass is HTMLLoader:
            loader = LoaderClass(
                source = raw_source,
                output_dir = args.output_dir,
                timeout=args.timeout,
                use_readability = args.use_readability
                )
        else:
            loader = LoaderClass(
                source = raw_source,
                output_dir = args.output_dir,
                timeout = args.timeout
                )
        loader.raw_content = raw
        loader.process()
