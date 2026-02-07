"""
Atlas-GRAG: Graph Retrieval Augmented Generation

Main entry point for the application.
"""

import logging
import sys

from src.config import get_config


def setup_logging(log_level: str) -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> int:
    """
    Main entry point for Atlas-GRAG.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    config = get_config()
    setup_logging(config.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Atlas-GRAG v0.1.0")
    logger.info(f"Debug mode: {config.debug}")
    
    # TODO: Initialize database connections
    # TODO: Load ingestion pipeline
    # TODO: Start Streamlit UI
    
    logger.info("Atlas-GRAG initialized successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
