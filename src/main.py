import sys
from pathlib import Path
from loguru import logger
import helper
from data_pipeline import DATA_DIR
sys.path.insert(0, str(Path(__file__).parent))

def main():   
    args = helper.Parser()
    helper.setup_logging()
    logger.info(f"Data directory: {DATA_DIR}")
    print(args)
    sources = args.sources
    if "all" in sources:
        sources = ["linux", "rust", "go", "networking", "books"]
    
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
        data_src = helper.ingest_all_data()
        if data_src:
            helper.create_training_dataset(data_src, args.output)
    logger.info("Data Loaded!")


if __name__ == "__main__":
    main()