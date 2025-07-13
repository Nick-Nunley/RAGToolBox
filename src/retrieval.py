import argparse
import os
import io
import requests
from urllib.parse import urlparse
import html2text
import pdfplumber
from Bio import Entrez


class BaseRetriever:
    """
    Abstract base class for fetching, converting, and saving content from a URL.
    """
    def __init__(self, url: str, output_dir: str):
        self.url = url
        self.output_dir = output_dir
        self.raw_content = None
        self.text = ""

    def fetch(self):
        """Fetch raw bytes from the URL."""
        response = requests.get(self.url)
        response.raise_for_status()
        self.raw_content = response.content

    @staticmethod
    def detect_retriever(url: str, content: bytes):
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

    def convert(self):
        """Convert raw content bytes to plain text. Implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement convert()")

    def save(self):
        """Save the converted text to a .txt file in the output directory."""
        os.makedirs(self.output_dir, exist_ok=True)
        parsed = urlparse(self.url)
        name = os.path.splitext(os.path.basename(parsed.path) or 'document')[0]
        filename = f"{name}.txt"
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(self.text)
        print(f"Saved plain text to {out_path}")

    def process(self):
        """Full pipeline: fetch, convert, and save."""
        print(f"Processing: {self.url}")
        self.fetch()
        self.convert()
        if self.text:
            self.save()
        else:
            print(f"Warning: No text extracted from {self.url}")

class NCBIRetriever(BaseRetriever):
    """
    Retriever that uses NCBI E-utilities (Entrez) to fetch abstracts or full text.
    """
    def __init__(self, url: str, output_dir: str):
        super().__init__(url, output_dir)
        self.pmc_id = os.path.basename(urlparse(self.url).path.rstrip('/'))

    def fetch(self):
        """Fetch content via Entrez efetch for PMC/PubMed IDs."""
        # Extract ID from URL (assumes last path segment is the PMC/PubMed ID)
        pmc_id = os.path.basename(urlparse(self.url).path.rstrip('/'))
        try:
            handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="abstract", retmode="text")
            text = handle.read()
            handle.close()
            self.raw_content = text.encode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Entrez fetch failed for {pmc_id}: {e}")

    def convert(self):
        """Convert the fetched Entrez text to plain text."""
        self.text = self.raw_content.decode('utf-8', errors='ignore')

    def save(self):
        """Save the converted text using PMC ID as filename."""
        os.makedirs(self.output_dir, exist_ok=True)
        filename = f"{self.pmc_id}.txt"
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(self.text)
        print(f"Saved plain text to {out_path}")

class HTMLRetriever(BaseRetriever):

    def convert(self):
        """Convert HTML content to markdown-like plain text."""
        self.text = html2text.html2text(
            self.raw_content.decode('utf-8', errors = 'ignore')
        )

class PDFRetriever(BaseRetriever):

    def convert(self):
        """Extract text from each page of a PDF."""
        text_chunks = []
        with pdfplumber.open(io.BytesIO(self.raw_content)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    text_chunks.append(txt)
        self.text = "\n".join(text_chunks)

class TextRetriever(BaseRetriever):

    def convert(self):
        """Decode raw bytes as UTF-8 text."""
        self.text = self.raw_content.decode('utf-8', errors = 'ignore')

class UnknownRetriever(BaseRetriever):

    def convert(self):
        """Handle unknown formats gracefully by skipping."""
        print(f"Unknown format for URL: {self.url}")
        self.text = ''


def main():
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
    args = parser.parse_args()

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
