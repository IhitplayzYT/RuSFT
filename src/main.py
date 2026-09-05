"""
Main orchestration script for pulling and preparing SFT training data.
This script coordinates data pulling from various sources and ingestion for LLM fine-tuning.
"""

import sys
from pathlib import Path
from loguru import logger
import helper
# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from data_pipeline import DATA_DIR

def main():   
    args = helper.Parser()
    helper.setup_logging()
    
    logger.info("Starting data preparation for LLM fine-tuning")
    logger.info(f"Data directory: {DATA_DIR}")
    
    # Determine which sources to pull
    sources = args.sources
    if "all" in sources:
        sources = ["linux", "rust", "go", "networking", "books"]
    
    # Pull data
    if not args.skip_pull:
        if "linux" in sources:
            helper.pull_linux_data()
        
        if "rust" in sources:
            helper.pull_rust_data()
        
        if "go" in sources:
            helper.pull_go_data()
        
        if "networking" in sources:
            helper.pull_networking_data(rfc_count=args.rfc_count)
        
        if "books" in sources or args.external_books:
            helper.process_external_books(args.external_books)
    
    if not args.skip_ingest:
        data_sources = helper.ingest_all_data()
        
        if data_sources:
            helper.create_training_dataset(data_sources, args.output)
    
    logger.info("=" * 100)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("=" * 100)

    



if __name__ == "__main__":
    main()