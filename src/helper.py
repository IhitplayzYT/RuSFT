
from loguru import logger
import argparse
from pathlib import Path
from data_pipeline import (
    LinuxDataPuller,
    RustDataPuller,
    GoDataPuller,
    NetworkingDataPuller,
    BookDataProcessor,
    DataAggregator,
    DataIngestor,
    DATA_DIR
)
import sys

def Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["linux", "rust", "go", "networking", "books", "all"],
        default=["all"],
        help="Data sources to pull (default: all)"
    )
    parser.add_argument(
        "--rfc-count",
        type=int,
        default=50,
        help="Number of RFC documents to download (default: 50)"
    )
    parser.add_argument(
        "--external-books",
        type=Path,
        default=None,
        help="Path to external directory containing book PDFs"
    )
    parser.add_argument(
        "--skip-pull",
        action="store_true",
        help="Skip data pulling, only ingest existing data"
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip data ingestion, only pull data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for training dataset (default: data/training_dataset.jsonl)"
    )
    return parser
    
def setup_logging():
    logger.remove()
    logger.add(sys.stdout,format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",level="INFO")
    logger.add(DATA_DIR / "data_preparation.log",format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",level="DEBUG",rotation="10 MB")


def pull_linux_data():
    try:
        kernel_path = LinuxDataPuller.clone_linux_kernel(branch="master", depth=1)
        logger.info(f"[SUCCESS] Linux kernel source: {kernel_path}")
    except Exception as e:
        logger.error(f"[FAILED] Failed to clone Linux kernel: {e}")
    
    try:
        man_pages_path = LinuxDataPuller.clone_man_pages()
        logger.info(f"[SUCCESS] Man pages: {man_pages_path}")
    except Exception as e:
        logger.error(f"[FAILED] Failed to clone man pages: {e}")
    
    try:
        docs_path = LinuxDataPuller.download_linux_docs()
        logger.info(f"[SUCCESS] Linux documentation: {docs_path}")
    except Exception as e:
        logger.error(f"[FAILED] Failed to download Linux docs: {e}")


def pull_rust_data():
    
    try:
        rust_path = RustDataPuller.clone_rust_std(branch="master", depth=1)
        logger.info(f"[SUCCESS] Rust source: {rust_path}")
    except Exception as e:
        logger.error(f"[FAILED] Failed to clone Rust source: {e}")
    
    try:
        docs_path = RustDataPuller.download_rust_docs()
        logger.info(f"[SUCCESS] Rust documentation: {docs_path}")
    except Exception as e:
        logger.error(f"[FAILED] Failed to download Rust docs: {e}")


def pull_go_data():
    try:
        go_path = GoDataPuller.clone_go_source(branch="master", depth=1)
        logger.info(f"[SUCCESS] Go source: {go_path}")
    except Exception as e:
        logger.error(f"[FAILED] Failed to clone Go source: {e}")
    
    try:
        docs_path = GoDataPuller.download_go_docs()
        logger.info(f"[SUCCESS] Go documentation: {docs_path}")
    except Exception as e:
        logger.error(f"[FAILED] Failed to download Go docs: {e}")
    
    try:
        wiki_path = GoDataPuller.scrape_go_wiki()
        logger.info(f"[SUCCESS] Go wiki: {wiki_path}")
    except Exception as e:
        logger.error(f"[FAILED] Failed to scrape Go wiki: {e}")


def pull_networking_data(rfc_count: int = 150):
    try:
        rfc_path = NetworkingDataPuller.download_rfcs(count=rfc_count)
        logger.info(f"[SUCCESS] RFC documents: {rfc_path}")
    except Exception as e:
        logger.error(f"[FAILED] Failed to download RFCs: {e}")
    
    try:
        books_path = NetworkingDataPuller.download_networking_books()
        logger.info(f"[SUCCESS] Networking books: {books_path}")
    except Exception as e:
        logger.error(f"[FAILED] Failed to download networking books: {e}")


def process_external_books(external_dir: Path = None):
    
    try:
        processed_path = BookDataProcessor.process_external_books(external_dir)
        logger.info(f"[SUCCESS] Processed books: {processed_path}")
    except Exception as e:
        logger.error(f"[FAILED] Failed to process external books: {e}")


def ingest_all_data():
    try:
        data_sources = DataAggregator.aggregate_all_data()
        logger.info(f"[SUCCESS] Aggregated {len(data_sources)} data sources")
        
        for source, files in data_sources.items():
            logger.info(f"  {source}: {len(files)} files")
        return data_sources
    except Exception as e:
        logger.error(f"[FAILED] Failed to aggregate data: {e}")
        return None


def create_training_dataset(data_sources, output_path: Path = None):
    if output_path is None:
        output_path = DATA_DIR / "training_dataset.jsonl"
    try:
        dataset_path = DataAggregator.create_training_dataset(data_sources, output_path)
        logger.info(f"[SUCCESS] Training dataset: {dataset_path}")
        return dataset_path
    except Exception as e:
        logger.error(f"[FAILED] Failed to create training dataset: {e}")
        return None

