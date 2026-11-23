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
import pymupdf 
from docx import Document

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
            doc = pymupdf.open(path)
            ret = ""
            for page in doc:
                ret += page.get_text()
            doc.close()
            return ret
        except Exception as e:
                logger.error(f"Failed to read pdf {path}: {e}")
                return ""
    
    @staticmethod
    def ingest_md(md_path: Path) -> str:
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                buff = f.read()
            return buff 
        except Exception as e:
            logger.error(f"Failed to read md file {md_path}: {e}")
            return ""
    
    @staticmethod
    def ingest_text(txt_path: Path) -> str:
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                buff = f.read()
                return buff
        except Exception as e:
            logger.error(f"Failed to read txt file {txt_path}: {e}")
            return ""
    
    @staticmethod
    def ingest_html(html_path: Path) -> str:
        try:
            with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            soup = BeautifulSoup(content, 'lxml')
            # Remove script and style elems
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator='\n', strip=True)
        except Exception as e:
            logger.error(f"Failed to read HTML {html_path}: {e}")
            return ""
    
    @staticmethod
    def ingest_docx(docx_path: Path) -> str:
        try:
            doc = Document(docx_path)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Failed to read Doc {docx_path}: {e}")
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
            return cls.ingest_text(file_path)
        else:
            logger.warning(f"Unknown file type: {suffix}, trying as text")
            return cls.ingest_text(file_path)


class LinuxDataPuller:
    """Pull Kernel docs and source code."""
    
    LINUX_KERNEL_REPO = "https://github.com/torvalds/linux.git"
    LINUX_DOCS_URL = "https://www.kernel.org/doc/html/latest/"
    MAN_PAGES_REPO = "https://git.kernel.org/pub/scm/docs/man-pages/man-pages.git"
    
    @staticmethod
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=5))
    def clone_linux_kernel(branch: str = "master", depth: int = 1) -> Path:
        save_pth = LINUX_DIR / "kernel-source"
        if save_pth.exists():
            return save_pth
        
        logger.info("Cloning Linux kernel")

        try:
            Repo.clone_from(LinuxDataPuller.LINUX_KERNEL_REPO,save_pth,branch=branch,depth=depth,single_branch=True)
            logger.info(f"Linux kernel saved to {save_pth}")
            return save_pth
        except Exception as e:
            logger.error(f"Failed to clone Linux kernel: {e}")
            raise
    
    @staticmethod
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=5))
    def clone_man_pages() -> Path:
        save_pth = LINUX_DIR / "man_pages"
        if save_pth.exists():
            return save_pth
        logger.info("Cloning Linux man pages")
        try:
            Repo.clone_from(LinuxDataPuller.MAN_PAGES_REPO, save_pth)
            logger.info(f"Man pages cloned to {save_pth}")
            return save_pth
        except Exception as e:
            logger.error(f"Failed to clone man pages: {e}")
            raise
    
    @staticmethod
    def download_linux_docs() -> Path:
        save_pth = LINUX_DIR / "documentation"
        save_pth.mkdir(exist_ok=True)
        logger.info(f"Downloading Linux docs")
        try:
            subprocess.run(["wget","--mirror","--convert-links","--adjust-extension","--page-requisites","--no-parent","--directory-prefix=" + str(save_pth),LinuxDataPuller.LINUX_DOCS_URL], check=True)
            logger.info(f"Linux docs downloaded to {save_pth}")
            return save_pth
        except (subprocess.CalledProcessError, Exception) as e:
            logger.error(f"Failed to download Linux docs: {e}")
            LinuxDataPuller.download_key_linux_docs(save_pth)
            return save_pth
    
    @staticmethod
    def download_key_linux_docs(docs_dir: Path):
        key_docs = ["https://www.kernel.org/doc/html/latest/process/","https://www.kernel.org/doc/html/latest/admin-guide/","https://www.kernel.org/doc/html/latest/networking/"]
        names = [k.split("/")[-2] for k in key_docs]
        for i,doc_url in enumerate(key_docs):
            try:
                resp = requests.get(doc_url)
                resp.raise_for_status()
                fname = urlparse(doc_url).path.split('/')[-1] or f"{names[i]}_index.html"
                fpath = docs_dir / fname 
                with open(fpath, 'w', encoding='utf-8') as f:
                    f.write(resp.text)
                logger.info(f"Downloaded {doc_url}")
            except Exception as e:
                logger.error(f"Failed to download {doc_url}: {e}")


class RustDataPuller:
    RUST_STD_REPO = "https://github.com/rust-lang/rust.git"
    RUST_BOOK_URL = "https://doc.rust-lang.org/book/"
    RUST_BY_EXAMPLE_URL = "https://doc.rust-lang.org/rust-by-example/"
    
    @staticmethod
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=5))
    def clone_rust_std(branch: str = "master", depth: int = 1) -> Path:
        save_pth = RUST_DIR / "rust-source"
        if save_pth.exists():
            return save_pth
        logger.info("Cloning Rust source")
        try:
            Repo.clone_from(RustDataPuller.RUST_STD_REPO,save_pth,branch=branch,depth=depth,single_branch=True)
            logger.info(f"Rust src cloned to {save_pth}")
            return save_pth
        except Exception as e:
            logger.error(f"Failed to clone Rust src: {e}")
            raise
    
    @staticmethod
    def download_rust_docs() -> Path:
        save_pth = RUST_DIR / "documentation"
        save_pth.mkdir(exist_ok=True)
        logger.info("Downloading Rust docs")
        rust_docs = [("The Rust Book", RustDataPuller.RUST_BOOK_URL),("Rust by Example", RustDataPuller.RUST_BY_EXAMPLE_URL)]
        
        for name, url in rust_docs:
            try:
                RustDataPuller._download_doc_site(url, save_pth, name)
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")
        
        return save_pth
    
    @staticmethod
    def _download_doc_site(base_url: str, output_dir: Path, name: str):
        name = name.lower().replace(' ', '-').replace('_', '-')
        save_pth = output_dir / name 
        save_pth.mkdir(exist_ok=True)
        
        try:
            subprocess.run(["wget","--mirror","--convert-links","--adjust-extension","--page-requisites","--no-parent","--directory-prefix=" + str(save_pth),base_url], check=True)
            logger.info(f"Downloaded {name}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.info(f"Documentation downlaod failed: {e}")


class GoDataPuller:
    GO_REPO = "https://go.googlesource.com/go"
    GO_DOCS_URL = "https://go.dev/doc/"
    GO_EFFECTIVE_GO = "https://go.dev/doc/effective_go"
    
    @staticmethod
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=5))
    def clone_go_source(branch: str = "master", depth: int = 1) -> Path:
        save_pth = GO_DIR / "go-source"
        if save_pth.exists():
            return save_pth
        
        logger.info(f"Cloning Go source")
        try:
            Repo.clone_from(GoDataPuller.GO_REPO,save_pth,branch=branch,depth=depth,single_branch=True)
            logger.info(f"Go source cloned to {save_pth}")
            return save_pth
        except Exception as e:
            logger.error(f"Failed to clone Go source: {e}")
            raise
    
    @staticmethod
    def download_go_docs() -> Path:
        save_pth = GO_DIR / "documentation"
        save_pth.mkdir(exist_ok=True)

        logger.info(f"Downloading Go documentation to {save_pth}")
        go_resources = [("Go Documentation", GoDataPuller.GO_DOCS_URL),("Effective Go", GoDataPuller.GO_EFFECTIVE_GO)]
        for name, url in go_resources:
            try:
                GoDataPuller._download_go_resource(url, save_pth, name)
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")
        return save_pth
    
    @staticmethod
    def _download_go_resource(url: str, output_dir: Path, name: str):
        name = name.lower().replace(' ', '-')
        filepath = output_dir / f"{name}.html"
        resp = requests.get(url)
        resp.raise_for_status()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(resp.text)
        logger.info(f"Downloaded {name} to {filepath}")
    

class NetworkingDataPuller:
    RFC_INDEX_URL = "https://www.rfc-editor.org/rfc-index.html"
    RFC_BASE_URL = "https://www.rfc-editor.org/rfc/"
    
    @staticmethod
    def download_rfcs(rfc_numbers: Optional[List[int]] = None, count: int = 100) -> Path:
        """ Download RFC documents or n recent RFC documents """
        save_pth = NETWORKING_DIR / "rfcs"
        save_pth.mkdir(exist_ok=True)
        logger.info(f"Downloading RFC documents to {save_pth}")
        if rfc_numbers is None:
            rfc_numbers = NetworkingDataPuller._get_recent_rfc_numbers(count)
        for rfc in tqdm(rfc_numbers, desc="Downloading RFCs"):
            try:
                NetworkingDataPuller._download_rfc(rfc, save_pth)
            except Exception as e:
                logger.error(f"Failed to download RFC {rfc}: {e}")
        return save_pth
    
    @staticmethod
    def _get_recent_rfc_numbers(count: int) -> List[int]:
        start_rfc = 9500 # Recent ones
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
                    logger.info(f"Downloaded RFC {rfc_num}.{fmt}")
                    return
            except Exception:
                continue
        logger.warning(f"Couldn't download RFC {rfc_num}")
    
    @staticmethod
    def download_networking_books() -> Path:
        books_dir = NETWORKING_DIR / "books"
        books_dir.mkdir(exist_ok=True)    
        logger.info(f"Downloading networking books")
        resources = [("TCP/IP Guide", "https://www.tcpipguide.com/")]
        for name, url in resources:
            try:
                name = name.lower().replace(' ', '-').replace('_','-')
                resource_dir = books_dir / name 
                resource_dir.mkdir(exist_ok=True)
                subprocess.run(["wget","--mirror","--convert-links","--adjust-extension","--page-requisites","--no-parent","--directory-prefix=" + str(resource_dir),url], check=True)
                logger.info(f"Downloaded {name}")
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")
        return books_dir


class PDFDirProcessor:
    @staticmethod
    def process_dir(dir: Optional[Path] = None) -> Path:
        if dir is None:
            dir = BOOKS_DIR
        processed_dir = BOOKS_DIR / "processed"
        processed_dir.mkdir(exist_ok=True)
        logger.info(f"Processing books from {dir}")
        pdf_files = list(dir.rglob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {dir}")
            return processed_dir
        
        for pdf_path in tqdm(pdf_files, desc="Processing books"):
            try:
                save_pth = processed_dir / pdf_path.name
                shutil.copy2(pdf_path, save_pth)
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
        logger.info("Aggregating all data")
        srcs = {"linux": list(LINUX_DIR.rglob("*")),"rust": list(RUST_DIR.rglob("*")),"go": list(GO_DIR.rglob("*")),"networking": list(NETWORKING_DIR.rglob("*")),"books": list(BOOKS_DIR.rglob("*")),}        
        for src, files in srcs.items():
            srcs[src] = [f for f in files if f.is_file()]
        total_files = sum(len(files) for files in srcs.values())
        logger.info(f"Total files aggregated: {total_files}")
        for source, files in srcs.items():
            logger.info(f"{source}: {len(files)} files")
        return srcs 
    
    @staticmethod
    def create_training_dataset(srcs: Dict[str, List[Path]], src_pth: Path) -> Path:
        src_pth.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating training dataset")
        training_data = []
        for src, files in srcs.items():
            for file_path in tqdm(files, desc=f"Processing {src}"):
                try:
                    text = DataIngestor.ingest_file(file_path)
                    if text.strip():
                        training_data.append({"source": src,"file_path": str(file_path.relative_to(DATA_DIR)),"content": text,"token_count": len(text.split())})
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")

        with open(src_pth, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Training dataset saved with {len(training_data)} entries")
        summary_path = src_pth.parent / "dataset_summary.json"
        summary = {"total_entries": len(training_data),"total_tokens": sum(item["token_count"] for item in training_data),"sources": {source: len([f for f in files if f.is_file()]) for source, files in srcs.items()}}
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Dataset summary saved to {summary_path}")        
        return src_pth


def pull_all_data():
    try:
        LinuxDataPuller.clone_linux_kernel()
    except Exception as e:
        logger.error(f"Failed to clone Linux kernel: {e}")
    
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
        NetworkingDataPuller.download_rfcs(count=50)
    except Exception as e:
        logger.error(f"Failed to download RFCs: {e}")
    
    try:
        NetworkingDataPuller.download_networking_books()
    except Exception as e:
        logger.error(f"Failed to download networking books: {e}")
    
    logger.info("Data pull process completed!")


if __name__ == "__main__":
    pull_all_data()
