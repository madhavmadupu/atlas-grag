"""
Atlas-GRAG: Graph Retrieval Augmented Generation

Main entry point for the application.
Supports CLI commands for ingestion, querying, and launching the UI.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

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


def cmd_ingest(args: argparse.Namespace) -> int:
    """Handle the ingest command."""
    from src.ingestion.pipeline import IngestionPipeline
    
    logger = logging.getLogger(__name__)
    pipeline = IngestionPipeline()
    
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return 1
        
        logger.info(f"Ingesting file: {file_path}")
        result = asyncio.run(pipeline.ingest_file(file_path))
    else:
        logger.info("Ingesting sample data...")
        result = asyncio.run(pipeline.ingest_sample_data())
    
    logger.info(f"Ingestion complete:")
    logger.info(f"  Nodes created: {result.nodes_created}")
    logger.info(f"  Relationships: {result.relationships_created}")
    logger.info(f"  Documents: {result.documents_added}")
    
    if result.errors:
        logger.warning(f"  Errors: {len(result.errors)}")
        for error in result.errors[:5]:
            logger.warning(f"    - {error}")
    
    return 0 if result.success else 1


def cmd_query(args: argparse.Namespace) -> int:
    """Handle the query command."""
    from src.retriever.hybrid import HybridRetriever
    from src.llm.chains import ReasoningChain
    
    logger = logging.getLogger(__name__)
    
    retriever = HybridRetriever()
    chain = ReasoningChain()
    
    logger.info(f"Query: {args.question}")
    
    # Retrieve context
    result = retriever.retrieve(args.question, include_graph=True)
    
    if result.entities:
        logger.info(f"Entities found: {', '.join(result.entities)}")
    
    # Generate answer
    response = chain.reason(result, args.question)
    
    print("\n" + "=" * 50)
    print("ANSWER:")
    print("=" * 50)
    print(response.answer)
    
    if args.verbose and response.reasoning:
        print("\n" + "-" * 50)
        print("REASONING:")
        print("-" * 50)
        print(response.reasoning)
    
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """Launch the Streamlit UI."""
    import subprocess
    
    logger = logging.getLogger(__name__)
    logger.info("Launching Streamlit UI...")
    
    app_path = Path(__file__).parent / "src" / "app" / "main.py"
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.headless", "true" if args.headless else "false",
    ])
    
    return 0


def main() -> int:
    """
    Main entry point for Atlas-GRAG.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = argparse.ArgumentParser(
        description="Atlas-GRAG: Graph Retrieval Augmented Generation for Supply Chain Risk Analysis"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest documents into the knowledge base"
    )
    ingest_parser.add_argument(
        "--file", "-f", type=str, help="Path to file to ingest (default: sample data)"
    )
    
    # Query command
    query_parser = subparsers.add_parser(
        "query", help="Query the knowledge base"
    )
    query_parser.add_argument(
        "question", type=str, help="Question to ask"
    )
    query_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show reasoning trace"
    )
    
    # UI command
    ui_parser = subparsers.add_parser(
        "ui", help="Launch the Streamlit dashboard"
    )
    ui_parser.add_argument(
        "--headless", action="store_true", help="Run in headless mode"
    )
    
    args = parser.parse_args()
    
    # Setup
    config = get_config()
    log_level = "DEBUG" if args.debug else config.log_level
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Atlas-GRAG v0.1.0")
    
    # Route to command handler
    if args.command == "ingest":
        return cmd_ingest(args)
    elif args.command == "query":
        return cmd_query(args)
    elif args.command == "ui":
        return cmd_ui(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

