import argparse
import os
import io
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import html2text
from readability import Document
import re
import pdfplumber
from Bio import Entrez
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any, List, Union


class BaseRetriever:
    """
    Abstract base class for fetching, converting, and saving content from a URL.
    """
    url: str
    output_dir: str
    raw_content: Optional[Union[bytes, str]]
    text: str

    def __init__(self, url: str, output_dir: str) -> None:
        self.url = url
        self.output_dir = output_dir
        self.raw_content = None
        self.text = ""

    def fetch(self) -> None:
        """Fetch raw bytes from the URL."""
        response = requests.get(self.url)
        response.raise_for_status()
        self.raw_content = response.content

    @staticmethod
    def detect_retriever(url: str, content: bytes) -> type:
        """
        Factory method to select the appropriate Retriever subclass.
        """
        path = urlparse(url).path.lower()
        ext = os.path.splitext(path)[1]
        head = content[:512].lstrip().lower()

        if 'ncbi.nlm.nih.gov' in urlparse(url).netloc:
            return NCBIRetriever
        if ext in ['.html', '.htm'] or head.startswith(b'<html') or head.startswith(b'<!doctype html'):
            return HTMLRetriever
        if ext == '.pdf':
            return PDFRetriever
        if ext in ['.txt', '.md']:
            return TextRetriever
        return UnknownRetriever

    def convert(self) -> None:
        """Convert raw content bytes to plain text. Implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement convert()")

    def save(self) -> None:
        """Save the converted text to a .txt file in the output directory."""
        os.makedirs(self.output_dir, exist_ok=True)
        parsed = urlparse(self.url)
        name = os.path.splitext(os.path.basename(parsed.path) or 'document')[0]
        filename = f"{name}.txt"
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(self.text)
        print(f"Saved plain text to {out_path}")

    def process(self) -> None:
        """Full pipeline: fetch, convert, and save."""
        print(f"Processing: {self.url}")
        self.fetch()
        self.convert()
        if self.text:
            self.save()
        else:
            print(f"Warning: No text extracted from {self.url}")

class NCBIRetriever(BaseRetriever):
    pmc_id: str
    _used_pdf: bool
    article_data: Optional[Dict[str, Any]]

    def __init__(self, url: str, output_dir: str) -> None:
        super().__init__(url, output_dir)
        self.pmc_id = os.path.basename(urlparse(self.url).path.rstrip('/'))
        self._used_pdf = False
        self.article_data = None

    def _get_ncbi_db_from_url(self) -> str:
        netloc = urlparse(self.url).netloc
        if 'pmc.' in netloc:
            return 'pmc'
        elif 'pubmed.' in netloc:
            return 'pubmed'
        else:
            return 'pmc'

    def fetch(self) -> None:
        """Fetch content via Entrez efetch for PMC/PubMed IDs, prefer PDF if available."""
        pmc_id = os.path.basename(urlparse(self.url).path.rstrip('/'))
        db = self._get_ncbi_db_from_url()
        tried: List[str] = []
        # Fetch XML from the correct db
        try:
            handle = Entrez.efetch(db=db, id=pmc_id, rettype="full" if db == "pmc" else "xml", retmode="xml")
            xml_content = handle.read()
            handle.close()
            self.raw_content = xml_content
        except Exception as e:
            tried.append(f"{db}/xml: {e}")
            raise RuntimeError(f"Entrez fetch failed for {pmc_id}: {tried}")

        # Try to extract PDF link if PMC
        pdf_url = self._extract_pdf_url_from_xml(self.raw_content) if db == 'pmc' else None
        if pdf_url:
            print(f"PDF link found for {pmc_id}: {pdf_url}\nAttempting to download and extract text from PDF.")
            try:
                pdf_bytes = self._download_pdf(pdf_url)
                pdf_retriever = PDFRetriever(self.url, self.output_dir)
                pdf_retriever.raw_content = pdf_bytes
                pdf_retriever.convert()
                self.text = pdf_retriever.text
                self._used_pdf = True
                return
            except Exception as e:
                print(f"Failed to download or process PDF for {pmc_id}: {e}\nFalling back to XML extraction.")
                self._used_pdf = False
        else:
            self._used_pdf = False

        # If no PDF, check for 'not allowed' comment and warn (handle both bytes and str)
        raw_str: str = self.raw_content.decode('utf-8', errors='ignore') if isinstance(self.raw_content, bytes) else self.raw_content
        if "does not allow downloading of the full text" in raw_str:
            print(f"Warning: Full text not available for {pmc_id}. Only abstract and metadata will be extracted.")

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
                        else:
                            pmc_id = os.path.basename(urlparse(self.url).path.rstrip('/'))
                            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/{href}"
            return None
        except Exception as e:
            print(f"Error parsing XML for PDF link: {e}")
            return None

    def _download_pdf(self, pdf_url: str) -> bytes:
        response = requests.get(pdf_url)
        response.raise_for_status()
        return response.content

    def safe_text(self, elem: Optional[ET.Element]) -> str:
        return elem.text.strip() if elem is not None and elem.text else ""

    def extract_all_text(self, elem: Optional[ET.Element]) -> str:
        if elem is None:
            return ""
        texts: List[str] = []
        if elem.text:
            texts.append(elem.text.strip())
        for child in elem:
            texts.append(self.extract_all_text(child))
        if elem.tail:
            texts.append(elem.tail.strip())
        return " ".join([t for t in texts if t])

    def convert(self) -> None:
        if getattr(self, '_used_pdf', False) and self.text:
            return
        self.article_data = self._parse_xml_content()
        if not self.article_data:
            self.text = ""
            return
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

    def _parse_xml_content(self) -> Optional[Dict[str, Any]]:
        """Dynamically parse PMC or PubMed XML content."""
        try:
            root = ET.fromstring(self.raw_content)
            tag = root.tag.lower()
            # Detect PMC or PubMed XML
            if tag.endswith('pmc-articleset') or tag.endswith('article'):
                return self._parse_pmc_xml(root)
            elif tag.endswith('pubmedarticleset') or tag.endswith('pubmedarticle'):
                return self._parse_pubmed_xml(root)
            else:
                print(f"Unknown XML root tag: {tag}. Returning empty article data.")
                return {}
        except Exception as e:
            print(f"Error parsing XML content: {e}")
            return None

    def _parse_pmc_xml(self, root: ET.Element) -> Dict[str, Any]:
        article_data: Dict[str, Any] = {}
        # Title
        title_elem = root.find(".//article-title")
        article_data['title'] = self.safe_text(title_elem)
        # Authors
        authors_list: List[str] = []
        for author_elem in root.findall(".//contrib[@contrib-type='author']"):
            surname = author_elem.find(".//surname")
            given_names = author_elem.find(".//given-names")
            if surname is not None and given_names is not None:
                authors_list.append(f"{self.safe_text(given_names)} {self.safe_text(surname)}")
        article_data['authors'] = authors_list
        # Journal
        journal_elem = root.find(".//journal-title")
        article_data['journal'] = self.safe_text(journal_elem)
        # DOI
        doi_elem = root.find(".//article-id[@pub-id-type='doi']")
        article_data['doi'] = self.safe_text(doi_elem)
        # Keywords
        keywords_list: List[str] = []
        for keyword_elem in root.findall(".//kwd"):
            kw = self.safe_text(keyword_elem)
            if kw:
                keywords_list.append(kw)
        article_data['keywords'] = keywords_list
        # Abstract
        abstract_elem = root.find(".//abstract")
        article_data['abstract'] = self.extract_all_text(abstract_elem)
        # Body
        body_elem = root.find(".//body")
        article_data['body'] = self.extract_all_text(body_elem)
        if not article_data['body']:
            print(f"Warning: Full text/body not available for {self.pmc_id}. Only abstract and metadata will be extracted.")
        # References
        references_list: List[str] = []
        for ref_elem in root.findall(".//ref"):
            ref_text = self.extract_all_text(ref_elem)
            if ref_text:
                references_list.append(ref_text)
        article_data['references'] = "\n".join(references_list)
        return article_data

    def _parse_pubmed_xml(self, root: ET.Element) -> Dict[str, Any]:
        article_data: Dict[str, Any] = {}
        # Title
        title_elem = root.find(".//ArticleTitle")
        article_data['title'] = self.safe_text(title_elem)
        # Authors
        authors_list: List[str] = []
        for author_elem in root.findall(".//AuthorList/Author"):
            last = self.safe_text(author_elem.find("LastName"))
            fore = self.safe_text(author_elem.find("ForeName"))
            if last or fore:
                authors_list.append(f"{fore} {last}".strip())
        article_data['authors'] = authors_list
        # Journal
        journal_elem = root.find(".//Journal/Title")
        article_data['journal'] = self.safe_text(journal_elem)
        # DOI
        doi_elem = root.find(".//ELocationID[@EIdType='doi']")
        article_data['doi'] = self.safe_text(doi_elem)
        # Keywords (PubMed Mesh terms)
        keywords_list: List[str] = []
        for mesh_elem in root.findall(".//MeshHeading/DescriptorName"):
            kw = self.safe_text(mesh_elem)
            if kw:
                keywords_list.append(kw)
        article_data['keywords'] = keywords_list
        # Abstract
        abstract_elem = root.find(".//Abstract")
        article_data['abstract'] = self.extract_all_text(abstract_elem)
        # Body (not available in PubMed, leave blank)
        article_data['body'] = ""
        # References
        references_list: List[str] = []
        for ref_elem in root.findall(".//ReferenceList/Reference/Citation"):
            ref_text = self.extract_all_text(ref_elem)
            if ref_text:
                references_list.append(ref_text)
        article_data['references'] = "\n".join(references_list)
        return article_data

    def _clean_text(self, text: str) -> str:
        """Clean up text by removing common HTML tags and extra whitespace."""
        text = re.sub(r'<[^>]+>', '', text) # Remove all HTML tags
        text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with single space
        text = text.strip()
        return text

    def save(self) -> None:
        """Save the converted text using PMC ID as filename."""
        os.makedirs(self.output_dir, exist_ok=True)
        filename = f"{self.pmc_id}.txt"
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(self.text)
        print(f"Saved plain text to {out_path}")

class HTMLRetriever(BaseRetriever):

    def __init__(self, url, output_dir, use_readability: bool = False):
        super().__init__(url, output_dir)
        self._USE_READABILITY = use_readability

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
                return el
        # HTML5 semantic
        for tag in ("article", "main"):
            el = soup.find(tag)
            if el and el.find("p"):
                return el
        # Fallback on the whole soup
        return soup

    def _clean_lines(self, text_block: str) -> str:
        return "\n".join(line.strip() for line in text_block.splitlines() if line.strip())

    def convert(self) -> None:
        """Extract only the main article text + references, generically across sites."""
        html = self.raw_content
        soup = BeautifulSoup(html, "html.parser")
        # 1) Find the best container
        main = self._find_main_container(soup)
        # 2) Readability fallback
        text_blob = main.get_text("\n", strip=True)
        if len(text_blob) < 200 and self._USE_READABILITY:
            doc = Document(html.decode("utf-8", errors="ignore"))
            soup = BeautifulSoup(doc.summary(), "html.parser")
            main = soup
        # 3) Pull text and strip UI cruft
        raw = main.get_text("\n", strip=True)
        raw = re.sub(r"\[\s*edit(?: on [^\]]*)?\s*\]", "", raw, flags=re.IGNORECASE)
        # 4) Split and clean body + references
        body, refs = raw.split("\nReferences", 1) if "\nReferences" in raw else (raw, "")
        clean = self._clean_lines(body)
        if refs:
            clean += "\n\nReferences\n" + self._clean_lines(refs)
        # 5) Markdown conversion (optional)
        self.text = html2text.html2text(clean)

    def _extract_title(self, html: bytes) -> str:
        soup = BeautifulSoup(html, "html.parser")
        # 1) try citation_title
        meta = soup.find("meta", {"name": "citation_title"})
        if meta and meta.get("content"):
            return meta["content"]
        # 2) try OpenGraph
        og = soup.find("meta", {"property": "og:title"})
        if og and og.get("content"):
            return og["content"]
        # 3) fallback to <title>
        if soup.title and soup.title.string:
            return soup.title.string
        return ""

    def save(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        # Try extracting a real title:
        title = self._extract_title(self.raw_content)  
        if title:
            name = self._slugify(title)
        else:
            # Fallback to base class logic
            parsed = urlparse(self.url)
            basename = os.path.basename(parsed.path) or "document"
            name = os.path.splitext(basename)[0]
        filename = f"{name}.txt"
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(self.text)
        print(f"Saved plain text to {out_path}")

class PDFRetriever(BaseRetriever):

    def convert(self) -> None:
        """Extract text from each page of a PDF."""
        text_chunks = []
        with pdfplumber.open(io.BytesIO(self.raw_content)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    text_chunks.append(txt)
        self.text = "\n".join(text_chunks)

class TextRetriever(BaseRetriever):

    def convert(self) -> None:
        """Decode raw bytes as UTF-8 text."""
        self.text = self.raw_content.decode('utf-8', errors = 'ignore')

class UnknownRetriever(BaseRetriever):

    def convert(self) -> None:
        """Handle unknown formats gracefully by skipping."""
        print(f"Unknown format for URL: {self.url}")
        self.text = ''


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch multiple URLs and convert content to plain text for chunking"
    )
    parser.add_argument(
        'urls', nargs='+', help='One or more URLs to ingest'
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
    args = parser.parse_args()

    HTMLRetriever._USE_READABILITY = args.use_readability
    # Set Entrez email if provided
    if args.email:
        Entrez.email = args.email
    else:
        if not Entrez.email:
            print("Warning: No email provided for NCBI E-utilities; they may block requests.")

    for url in args.urls:
        try:
            raw = requests.get(url).content
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            continue

        RetrieverClass = BaseRetriever.detect_retriever(url, raw)
        retriever = RetrieverClass(url, args.output_dir)
        retriever.raw_content = raw
        retriever.process()

if __name__ == '__main__':

    main()
