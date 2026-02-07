"""
Tests for the configuration module.
"""

import os
from pathlib import Path

import pytest

from src.config import (
    AppConfig,
    ChromaConfig,
    Neo4jConfig,
    OllamaConfig,
    RetrievalConfig,
    get_config,
)


class TestNeo4jConfig:
    """Tests for Neo4j configuration."""
    
    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        config = Neo4jConfig()
        
        assert config.uri == "bolt://localhost:7687"
        assert config.username == "neo4j"
        assert config.database == "neo4j"
    
    def test_respects_environment_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should read from environment variables."""
        monkeypatch.setenv("NEO4J_URI", "bolt://custom:7687")
        monkeypatch.setenv("NEO4J_USERNAME", "custom_user")
        
        config = Neo4jConfig()
        
        assert config.uri == "bolt://custom:7687"
        assert config.username == "custom_user"


class TestChromaConfig:
    """Tests for ChromaDB configuration."""
    
    def test_default_persist_directory(self) -> None:
        """Should have default persist directory."""
        config = ChromaConfig()
        
        assert config.persist_directory == Path("./data/chroma")
    
    def test_default_collection_name(self) -> None:
        """Should have default collection name."""
        config = ChromaConfig()
        
        assert config.collection_name == "supply_chain_docs"


class TestOllamaConfig:
    """Tests for Ollama configuration."""
    
    def test_default_base_url(self) -> None:
        """Should have default Ollama base URL."""
        config = OllamaConfig()
        
        assert config.base_url == "http://localhost:11434"
    
    def test_default_model(self) -> None:
        """Should use llama3 as default model."""
        config = OllamaConfig()
        
        assert config.model == "llama3"
    
    def test_default_embedding_model(self) -> None:
        """Should use nomic-embed-text as default embedding model."""
        config = OllamaConfig()
        
        assert config.embedding_model == "nomic-embed-text"


class TestRetrievalConfig:
    """Tests for retrieval configuration."""
    
    def test_default_vector_top_k(self) -> None:
        """Should have default top-k value of 5."""
        config = RetrievalConfig()
        
        assert config.vector_top_k == 5
    
    def test_default_graph_max_hops(self) -> None:
        """Should have default max hops of 2."""
        config = RetrievalConfig()
        
        assert config.graph_max_hops == 2


class TestAppConfig:
    """Tests for main application configuration."""
    
    def test_contains_all_sub_configs(self) -> None:
        """Should contain all sub-configuration objects."""
        config = AppConfig()
        
        assert isinstance(config.neo4j, Neo4jConfig)
        assert isinstance(config.chroma, ChromaConfig)
        assert isinstance(config.ollama, OllamaConfig)
        assert isinstance(config.retrieval, RetrievalConfig)
    
    def test_default_log_level(self) -> None:
        """Should have INFO as default log level."""
        config = AppConfig()
        
        assert config.log_level == "INFO"
    
    def test_debug_disabled_by_default(self) -> None:
        """Should have debug disabled by default."""
        config = AppConfig()
        
        assert config.debug is False


class TestGetConfig:
    """Tests for the get_config function."""
    
    def test_returns_app_config(self) -> None:
        """Should return an AppConfig instance."""
        config = get_config()
        
        assert isinstance(config, AppConfig)
