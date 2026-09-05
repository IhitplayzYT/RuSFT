import os
import subprocess
import requests
from pathlib import Path
from typing import List, Optional, Dict 
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from git import Repo
import shutil

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LINUX_DIR = DATA_DIR / "linux"
RUST_DIR = DATA_DIR / "rust"
GO_DIR = DATA_DIR / "go"
NETWORKING_DIR = DATA_DIR / "networking"
BOOKS_DIR = DATA_DIR / "books"
PDFS_DIR = DATA_DIR / "pdfs"

for dir_path in [DATA_DIR, LINUX_DIR, RUST_DIR, GO_DIR, NETWORKING_DIR, BOOKS_DIR, PDFS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class DataIngestor:

    @staticmethod
    def ingest_pdf(path: Path) -> str:
        try:
            import pymupdf 
            doc = pymupdf.open(path)
            ret = ""
            for page in doc:
                ret += page.get_text()
            doc.close()
            return ret
        except Exception as e:
                logger.error(f"Failed to extract from {path}: {e}")
                return ""
    
    @staticmethod
    def ingest_markdown(md_path: Path) -> str:
        try:
            import markdown
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"Failed to read markdown {md_path}: {e}")
            return ""
    
    @staticmethod
    def ingest_text(txt_path: Path) -> str:
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read text file {txt_path}: {e}")
            return ""
    
    @staticmethod
    def ingest_html(html_path: Path) -> str:
        try:
            with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            soup = BeautifulSoup(content, 'lxml')
            for script in soup(["script", "style"]): # Remove script and style elems
                script.decompose()
            return soup.get_text(separator='\n', strip=True)
        except Exception as e:
            logger.error(f"Failed to read HTML {html_path}: {e}")
            return ""
    
    @staticmethod
    def ingest_docx(docx_path: Path) -> str:
        try:
            from docx import Document
            doc = Document(docx_path)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Failed to read DOCX {docx_path}: {e}")
            return ""
    
    @staticmethod
    def ingest_code(code_path: Path) -> str:
        try:
            with open(code_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read code file {code_path}: {e}")
            return ""
    
    @classmethod
    def ingest_file(cls, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            return cls.ingest_pdf(file_path)
        elif suffix in ['.md', '.markdown']:
            return cls.ingest_markdown(file_path)
        elif suffix in ['.txt', '.rst', '.asciidoc']:
            return cls.ingest_text(file_path)
        elif suffix in ['.html', '.htm']:
            return cls.ingest_html(file_path)
        elif suffix == '.docx':
            return cls.ingest_docx(file_path)
        elif suffix in ['.py', '.rs', '.go', '.c', '.cpp', '.h', '.js', '.ts', '.java', '.sh', '.bash']:
            return cls.ingest_code(file_path)
        else:
            logger.warning(f"Unknown file type: {suffix}, trying as text")
            return cls.ingest_text(file_path)


class LinuxDataPuller:
    """Pull Kernel documentation and source code."""
    
    LINUX_KERNEL_REPO = "https://github.com/torvalds/linux.git"
    LINUX_DOCS_URL = "https://www.kernel.org/doc/html/latest/"
    MAN_PAGES_REPO = "https://git.kernel.org/pub/scm/docs/man-pages/man-pages.git"
    
    @staticmethod
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    def clone_linux_kernel(branch: str = "master", depth: int = 1) -> Path:
        """Clone Linux kernel source code."""
        kernel_dir = LINUX_DIR / "kernel-source"
        
        if kernel_dir.exists():
            return kernel_dir
        
        logger.info(f"Cloning Linux kernel from {LinuxDataPuller.LINUX_KERNEL_REPO}")

        try:
            Repo.clone_from(LinuxDataPuller.LINUX_KERNEL_REPO,kernel_dir,branch=branch,depth=depth,single_branch=True)
            logger.info(f"Linux kernel cloned to {kernel_dir}")
            return kernel_dir
        except Exception as e:
            logger.error(f"Failed to clone Linux kernel: {e}")
            raise
    
    @staticmethod
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    def clone_man_pages() -> Path:
        """Clone Linux man pages."""
        man_dir = LINUX_DIR / "man_pages"
        
        if man_dir.exists():
            return man_dir
        
        logger.info(f"Cloning Linux man pages from {LinuxDataPuller.MAN_PAGES_REPO}")
        try:
            Repo.clone_from(LinuxDataPuller.MAN_PAGES_REPO, man_dir)
            logger.info(f"Man pages cloned to {man_dir}")
            return man_dir
        except Exception as e:
            logger.error(f"Failed to clone man pages: {e}")
            raise
    
    @staticmethod
    def download_linux_docs() -> Path:
        """Download Linux documentation from kernel.org."""
        docs_dir = LINUX_DIR / "documentation"
        docs_dir.mkdir(exist_ok=True)
        
        logger.info(f"Downloading Linux documentation to {docs_dir}")
        
        try:
            subprocess.run([
                "wget",
                "--mirror",
                "--convert-links",
                "--adjust-extension",
                "--page-requisites",
                "--no-parent",
                "--directory-prefix=" + str(docs_dir),
                LinuxDataPuller.LINUX_DOCS_URL
            ], check=True)

            logger.info(f"Linux docs downloaded to {docs_dir}")
            return docs_dir
        except subprocess.CalledProcessError and Exception as e:
            logger.error(f"Failed to download Linux docs: {e}")
            # Download imp docs
            LinuxDataPuller.download_key_linux_docs(docs_dir)
            return docs_dir
    
    @staticmethod
    def download_key_linux_docs(docs_dir: Path):
        key_docs = [
            "https://www.kernel.org/doc/html/latest/process/",
            "https://www.kernel.org/doc/html/latest/admin-guide/",
            "https://www.kernel.org/doc/html/latest/networking/",
        ]
        
        for doc_url in key_docs:
            try:
                response = requests.get(doc_url)
                response.raise_for_status()

                filename = urlparse(doc_url).path.split('/')[-1] or "index.html"
                filepath = docs_dir / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                logger.info(f"Downloaded {doc_url} to {filepath}")
            except Exception as e:
                logger.error(f"Failed to download {doc_url}: {e}")


class RustDataPuller:
    
    RUST_STD_REPO = "https://github.com/rust-lang/rust.git"
    RUST_BOOK_URL = "https://doc.rust-lang.org/book/"
    RUST_BY_EXAMPLE_URL = "https://doc.rust-lang.org/rust-by-example/"
    RUST_REFERENCE_URL = "https://doc.rust-lang.org/reference/"
    RUST_NOMICON_URL = "https://doc.rust-lang.org/nomicon/"
    RUST_STABLE_DOCS = "https://doc.rust-lang.org/stable/"
    
    @staticmethod
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    def clone_rust_std(branch: str = "master", depth: int = 1) -> Path:
        rust_dir = RUST_DIR / "rust-source"
        if rust_dir.exists():
            return rust_dir
        
        logger.info(f"Cloning Rust source from {RustDataPuller.RUST_STD_REPO}")
        try:
            Repo.clone_from(RustDataPuller.RUST_STD_REPO,rust_dir,branch=branch,depth=depth,single_branch=True)
            logger.info(f"Rust src cloned to {rust_dir}")
            return rust_dir
        except Exception as e:
            logger.error(f"Failed to clone Rust src: {e}")
            raise
    
    @staticmethod
    def download_rust_docs() -> Path:
        docs_dir = RUST_DIR / "documentation"
        docs_dir.mkdir(exist_ok=True)
        logger.info(f"Downloading Rust documentation to {docs_dir}")
        
        rust_docs = [
            ("The Rust Book", RustDataPuller.RUST_BOOK_URL),
            ("Rust by Example", RustDataPuller.RUST_BY_EXAMPLE_URL),
            ("The Rust Reference", RustDataPuller.RUST_REFERENCE_URL),
            ("The Rustonomicon", RustDataPuller.RUST_NOMICON_URL),
            ("Stable Docs", RustDataPuller.RUST_STABLE_DOCS),
        ]
        
        for name, url in rust_docs:
            try:
                RustDataPuller._download_doc_site(url, docs_dir, name)
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")
        
        return docs_dir
    
    @staticmethod
    def _download_doc_site(base_url: str, output_dir: Path, name: str):
        safe_name = name.lower().replace(' ', '-').replace('_', '-')
        site_dir = output_dir / safe_name
        site_dir.mkdir(exist_ok=True)
        
        try:
            subprocess.run([
                "wget",
                "--mirror",
                "--convert-links",
                "--adjust-extension",
                "--page-requisites",
                "--no-parent",
                "--directory-prefix=" + str(site_dir),
                base_url
            ], check=True)
            logger.info(f"Downloaded {name} to {site_dir}")
        except subprocess.CalledProcessError or FileNotFoundError as e:
            logger.info(f"Documentation downlaod failed: {e}")


class GoDataPuller:
    
    GO_REPO = "https://go.googlesource.com/go"
    GO_DOCS_URL = "https://go.dev/doc/"
    GO_EFFECTIVE_GO = "https://go.dev/doc/effective_go"
    GO_WIKI_REPO = "https://github.com/golang/go/wiki"
    
    @staticmethod
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    def clone_go_source(branch: str = "master", depth: int = 1) -> Path:
        go_dir = GO_DIR / "go-source"
        if go_dir.exists():
            return go_dir
        
        logger.info(f"Cloning Go source from {GoDataPuller.GO_REPO}")
        try:
            Repo.clone_from(GoDataPuller.GO_REPO,go_dir,branch=branch,depth=depth,single_branch=True)
            logger.info(f"Go source cloned to {go_dir}")
            return go_dir
        except Exception as e:
            logger.error(f"Failed to clone Go source: {e}")
            raise
    
    @staticmethod
    def download_go_docs() -> Path:
        docs_dir = GO_DIR / "documentation"
        docs_dir.mkdir(exist_ok=True)
        logger.info(f"Downloading Go documentation to {docs_dir}")
        go_resources = [
            ("Go Documentation", GoDataPuller.GO_DOCS_URL),
            ("Effective Go", GoDataPuller.GO_EFFECTIVE_GO),
        ]
        
        for name, url in go_resources:
            try:
                GoDataPuller._download_go_resource(url, docs_dir, name)
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")
        return docs_dir
    
    @staticmethod
    def _download_go_resource(url: str, output_dir: Path, name: str):
        safe_name = name.lower().replace(' ', '-')
        filepath = output_dir / f"{safe_name}.html"
        response = requests.get(url)
        response.raise_for_status()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        logger.info(f"Downloaded {name} to {filepath}")
    
    @staticmethod
    def scrape_go_wiki() -> Path:
        """Scrape Go wiki pages."""
        wiki_dir = GO_DIR / "wiki"
        wiki_dir.mkdir(exist_ok=True)
        
        logger.info("Scraping Go wiki")
        wiki_pages = [
            "https://github.com/golang/go/wiki",
            "https://github.com/golang/go/wiki/CodeReviewComments",
            "https://github.com/golang/go/wiki/Style",
            "https://github.com/golang/go/wiki/Performance",
        ]
        
        for page_url in wiki_pages:
            try:
                response = requests.get(page_url)
                response.raise_for_status()
                page_name = urlparse(page_url).path.split('/')[-1] or "index"
                filepath = wiki_dir / f"{page_name}.html"
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                logger.info(f"Downloaded wiki page {page_name}")
            except Exception as e:
                logger.error(f"Failed to download wiki page {page_url}: {e}")
        return wiki_dir


class NetworkingDataPuller:

    RFC_INDEX_URL = "https://www.rfc-editor.org/rfc-index.html"
    RFC_BASE_URL = "https://www.rfc-editor.org/rfc/"
    
    @staticmethod
    def download_rfcs(rfc_numbers: Optional[List[int]] = None, count: int = 100) -> Path:
        """ Download RFC documents or 'count' recent RFC documents """
        rfc_dir = NETWORKING_DIR / "rfcs"
        rfc_dir.mkdir(exist_ok=True)
        logger.info(f"Downloading RFC documents to {rfc_dir}")
        
        if rfc_numbers is None:
            rfc_numbers = NetworkingDataPuller._get_recent_rfc_numbers(count)
        for rfc_num in tqdm(rfc_numbers, desc="Downloading RFCs"):
            try:
                NetworkingDataPuller._download_rfc(rfc_num, rfc_dir)
            except Exception as e:
                logger.error(f"Failed to download RFC {rfc_num}: {e}")
        return rfc_dir
    
    @staticmethod
    def _get_recent_rfc_numbers(count: int) -> List[int]:
        # Start from a high number and work backwards
        # RFC 9000+ are recent (QUIC, HTTP/3, etc.)
        start_rfc = 9500
        return list(range(start_rfc, start_rfc - count, -1))
    
    @staticmethod
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _download_rfc(rfc_num: int, output_dir: Path):
        for fmt in ['txt', 'pdf']:
            url = f"{NetworkingDataPuller.RFC_BASE_URL}rfc{rfc_num}.{fmt}"
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    filepath = output_dir / f"rfc{rfc_num}.{fmt}"
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Downloaded RFC {rfc_num} ({fmt})")
                    return
            except Exception:
                continue
        
        logger.warning(f"Couldn't download RFC {rfc_num}")
    
    @staticmethod
    def download_networking_books() -> Path:
        """Download networking textbooks and resources."""
        books_dir = NETWORKING_DIR / "books"
        books_dir.mkdir(exist_ok=True)    
        logger.info(f"Downloading networking srcs to {books_dir}")
        
        resources = [
            ("TCP/IP Guide", "https://www.tcpipguide.com/"),
        ]
        
        for name, url in resources:
            try:
                safe_name = name.lower().replace(' ', '-').replace('_','-')
                resource_dir = books_dir / safe_name
                resource_dir.mkdir(exist_ok=True)
                subprocess.run([
                    "wget",
                    "--mirror",
                    "--convert-links",
                    "--adjust-extension",
                    "--page-requisites",
                    "--no-parent",
                    "--directory-prefix=" + str(resource_dir),
                    url
                ], check=True)
                
                logger.info(f"Downloaded {name}")
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")
        return books_dir


class BookDataProcessor:
    @staticmethod
    def process_external_books(external_dir: Optional[Path] = None) -> Path:
        if external_dir is None:
            external_dir = BOOKS_DIR
        
        processed_dir = BOOKS_DIR / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        logger.info(f"Processing books from {external_dir}")
        pdf_files = list(external_dir.rglob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {external_dir}")
            return processed_dir
        
        for pdf_path in tqdm(pdf_files, desc="Processing books"):
            try:
                dest_path = processed_dir / pdf_path.name
                shutil.copy2(pdf_path, dest_path)
                text = DataIngestor.ingest_pdf(pdf_path)
                text_path = processed_dir / f"{pdf_path.stem}.txt"
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                logger.info(f"Processed {pdf_path.name}")
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
        return processed_dir


class DataAggregator:
    @staticmethod
    def aggregate_all_data() -> Dict[str, List[Path]]:
        logger.info("Aggregating all training data")
        data_sources = {
            "linux": list(LINUX_DIR.rglob("*")),
            "rust": list(RUST_DIR.rglob("*")),
            "go": list(GO_DIR.rglob("*")),
            "networking": list(NETWORKING_DIR.rglob("*")),
            "books": list(BOOKS_DIR.rglob("*")),
        }
        for source, files in data_sources.items():
            data_sources[source] = [f for f in files if f.is_file()]
        total_files = sum(len(files) for files in data_sources.values())
        logger.info(f"Total files aggregated: {total_files}")
        for source, files in data_sources.items():
            logger.info(f"{source}: {len(files)} files")
        return data_sources
    
    @staticmethod
    def create_training_dataset(data_sources: Dict[str, List[Path]], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating training dataset at {output_path}")
        training_data = []
        for source, files in data_sources.items():
            for file_path in tqdm(files, desc=f"Processing {source}"):
                try:
                    text = DataIngestor.ingest_file(file_path)
                    if text.strip():
                        training_data.append({
                            "source": source,
                            "file_path": str(file_path.relative_to(DATA_DIR)),
                            "content": text,
                            "token_count": len(text.split())
                        })
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Training dataset saved with {len(training_data)} entries")
        summary_path = output_path.parent / "dataset_summary.json"
        summary = {
            "total_entries": len(training_data),
            "total_tokens": sum(item["token_count"] for item in training_data),
            "sources": {source: len([f for f in files if f.is_file()]) for source, files in data_sources.items()}
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Dataset summary saved to {summary_path}")        
        return output_path


def pull_all_data():
    logger.info("Pulling Linux data")
    try:
        LinuxDataPuller.clone_linux_kernel()
    except Exception as e:
        logger.error(f"Failed to clone Linux kernel: {e}")
    
    try:
        LinuxDataPuller.clone_man_pages()
    except Exception as e:
        logger.error(f"Failed to clone man pages: {e}")
    
    try:
        LinuxDataPuller.download_linux_docs()
    except Exception as e:
        logger.error(f"Failed to download Linux docs: {e}")
    
    logger.info("Pulling Rust data")
    try:
        RustDataPuller.clone_rust_std()
    except Exception as e:
        logger.error(f"Failed to clone Rust std: {e}")
    
    try:
        RustDataPuller.download_rust_docs()
    except Exception as e:
        logger.error(f"Failed to download Rust docs: {e}")
    
    logger.info("Pulling Go data")
    try:
        GoDataPuller.clone_go_source()
    except Exception as e:
        logger.error(f"Failed to clone Go source: {e}")
    
    try:
        GoDataPuller.download_go_docs()
    except Exception as e:
        logger.error(f"Failed to download Go docs: {e}")
    
    try:
        GoDataPuller.scrape_go_wiki()
    except Exception as e:
        logger.error(f"Failed to scrape Go wiki: {e}")
    
    logger.info("Pulling Networking data")
    try:
        NetworkingDataPuller.download_rfcs(count=50)
    except Exception as e:
        logger.error(f"Failed to download RFCs: {e}")
    
    try:
        NetworkingDataPuller.download_networking_books()
    except Exception as e:
        logger.error(f"Failed to download networking books: {e}")
    
    logger.info("Data pull process completed")


if __name__ == "__main__":
    pull_all_data()
