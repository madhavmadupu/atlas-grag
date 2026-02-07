"""
Atlas-GRAG Configuration Module.

Centralizes all configuration settings with environment variable support.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _get_env(key: str, default: Optional[str] = None) -> str:
    """Get environment variable with optional default."""
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} is not set and no default provided")
    return value


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


def _get_env_int(key: str, default: int) -> int:
    """Get integer environment variable."""
    return int(os.getenv(key, str(default)))


@dataclass(frozen=True)
class Neo4jConfig:
    """Neo4j database configuration."""
    
    uri: str = field(default_factory=lambda: _get_env("NEO4J_URI", "bolt://localhost:7687"))
    username: str = field(default_factory=lambda: _get_env("NEO4J_USERNAME", "neo4j"))
    password: str = field(default_factory=lambda: _get_env("NEO4J_PASSWORD", "password"))
    database: str = field(default_factory=lambda: _get_env("NEO4J_DATABASE", "neo4j"))


@dataclass(frozen=True)
class ChromaConfig:
    """ChromaDB configuration."""
    
    persist_directory: Path = field(
        default_factory=lambda: Path(_get_env("CHROMA_PERSIST_DIRECTORY", "./data/chroma"))
    )
    collection_name: str = field(
        default_factory=lambda: _get_env("CHROMA_COLLECTION_NAME", "supply_chain_docs")
    )


@dataclass(frozen=True)
class OllamaConfig:
    """Ollama LLM configuration."""
    
    base_url: str = field(
        default_factory=lambda: _get_env("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    model: str = field(default_factory=lambda: _get_env("OLLAMA_MODEL", "llama3"))
    embedding_model: str = field(
        default_factory=lambda: _get_env("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    )


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval settings."""
    
    vector_top_k: int = field(default_factory=lambda: _get_env_int("VECTOR_TOP_K", 5))
    graph_max_hops: int = field(default_factory=lambda: _get_env_int("GRAPH_MAX_HOPS", 2))


@dataclass(frozen=True)
class AppConfig:
    """Application-level configuration."""
    
    log_level: str = field(default_factory=lambda: _get_env("LOG_LEVEL", "INFO"))
    debug: bool = field(default_factory=lambda: _get_env_bool("DEBUG", False))
    
    # Sub-configurations
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


# Global configuration instance
def get_config() -> AppConfig:
    """Get the application configuration."""
    return AppConfig()
