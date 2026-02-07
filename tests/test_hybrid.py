"""
Tests for the Hybrid Retriever.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestHybridRetrieverInitialization:
    """Tests for HybridRetriever initialization."""

    def test_should_create_with_default_config(self) -> None:
        """Should create retriever with default configuration."""
        with patch("src.retriever.hybrid.GraphDatabaseManager") as mock_graph:
            with patch("src.retriever.hybrid.VectorDatabaseManager") as mock_vector:
                mock_graph_instance = MagicMock()
                mock_graph.return_value = mock_graph_instance
                
                mock_vector_instance = MagicMock()
                mock_vector.return_value = mock_vector_instance
                
                from src.retriever.hybrid import HybridRetriever
                
                retriever = HybridRetriever()
                
                assert retriever is not None


class TestEntityExtraction:
    """Tests for entity extraction from queries."""

    def test_should_extract_entities_from_query(self) -> None:
        """Should identify entity names in query text."""
        with patch("src.retriever.hybrid.GraphDatabaseManager") as mock_graph:
            with patch("src.retriever.hybrid.VectorDatabaseManager") as mock_vector:
                with patch("src.retriever.hybrid.OllamaLLM") as mock_llm_class:
                    mock_graph_instance = MagicMock()
                    mock_graph.return_value = mock_graph_instance
                    
                    mock_vector_instance = MagicMock()
                    mock_vector.return_value = mock_vector_instance
                    
                    mock_llm = MagicMock()
                    mock_llm_class.return_value = mock_llm
                    mock_llm.invoke.return_value = '["Singapore", "GlobalTech"]'
                    
                    from src.retriever.hybrid import HybridRetriever
                    
                    retriever = HybridRetriever()
                    entities = retriever._extract_entities(
                        "How will the strike in Singapore impact GlobalTech?"
                    )
                    
                    assert "Singapore" in entities or "globaltech" in entities.lower() if isinstance(entities, str) else any("singapore" in e.lower() or "globaltech" in e.lower() for e in entities)


class TestVectorRetrieval:
    """Tests for vector-based retrieval."""

    def test_should_retrieve_similar_documents(self) -> None:
        """Should retrieve semantically similar documents."""
        with patch("src.retriever.hybrid.GraphDatabaseManager") as mock_graph:
            with patch("src.retriever.hybrid.VectorDatabaseManager") as mock_vector:
                mock_graph_instance = MagicMock()
                mock_graph.return_value = mock_graph_instance
                
                mock_vector_instance = MagicMock()
                mock_vector.return_value = mock_vector_instance
                mock_vector_instance.query_similar.return_value = [
                    {"id": "doc1", "document": "Singapore port strike", "distance": 0.1},
                    {"id": "doc2", "document": "TechFlow manufactures in Singapore", "distance": 0.2}
                ]
                
                from src.retriever.hybrid import HybridRetriever
                
                retriever = HybridRetriever()
                docs = retriever._retrieve_vector("Singapore strike impact")
                
                assert len(docs) == 2


class TestGraphRetrieval:
    """Tests for graph-based retrieval."""

    def test_should_find_n_hop_neighbors(self) -> None:
        """Should find entities within N hops."""
        with patch("src.retriever.hybrid.GraphDatabaseManager") as mock_graph:
            with patch("src.retriever.hybrid.VectorDatabaseManager") as mock_vector:
                mock_graph_instance = MagicMock()
                mock_graph.return_value = mock_graph_instance
                # Mock execute_query which is used internally
                mock_graph_instance.execute_query.return_value = [
                    {"source": "Singapore", "target": "FlowChips", "relationships": ["OPERATES_AT"], "path_length": 1},
                    {"source": "Singapore", "target": "GlobalTech", "relationships": ["AFFECTS", "DEPENDS_ON"], "path_length": 2}
                ]
                
                mock_vector_instance = MagicMock()
                mock_vector.return_value = mock_vector_instance
                
                from src.retriever.hybrid import HybridRetriever
                
                retriever = HybridRetriever()
                neighbors = retriever._retrieve_graph_neighbors("Singapore")
                
                assert len(neighbors) >= 1

    def test_should_generate_cypher_query(self) -> None:
        """Should generate appropriate Cypher query for entities."""
        with patch("src.retriever.hybrid.GraphDatabaseManager") as mock_graph:
            with patch("src.retriever.hybrid.VectorDatabaseManager") as mock_vector:
                mock_graph_instance = MagicMock()
                mock_graph.return_value = mock_graph_instance
                
                mock_vector_instance = MagicMock()
                mock_vector.return_value = mock_vector_instance
                
                from src.retriever.hybrid import HybridRetriever
                
                retriever = HybridRetriever()
                query = retriever._build_neighbor_query("Singapore", max_hops=2)
                
                assert "MATCH" in query
                assert "Singapore" in query


class TestHybridSearch:
    """Tests for combined hybrid search."""

    def test_should_combine_vector_and_graph_results(self) -> None:
        """Should merge results from both retrieval methods."""
        with patch("src.retriever.hybrid.GraphDatabaseManager") as mock_graph:
            with patch("src.retriever.hybrid.VectorDatabaseManager") as mock_vector:
                with patch("src.retriever.hybrid.OllamaLLM") as mock_llm_class:
                    mock_graph_instance = MagicMock()
                    mock_graph.return_value = mock_graph_instance
                    mock_graph_instance.find_neighbors.return_value = [
                        {"node": {"name": "FlowChips"}, "path_length": 1}
                    ]
                    mock_graph_instance.execute_query.return_value = [
                        {"name": "Singapore"}
                    ]
                    
                    mock_vector_instance = MagicMock()
                    mock_vector.return_value = mock_vector_instance
                    mock_vector_instance.query_similar.return_value = [
                        {"id": "doc1", "document": "Singapore strike", "distance": 0.1}
                    ]
                    
                    mock_llm = MagicMock()
                    mock_llm_class.return_value = mock_llm
                    mock_llm.invoke.return_value = '["Singapore"]'
                    
                    from src.retriever.hybrid import HybridRetriever
                    
                    retriever = HybridRetriever()
                    result = retriever.retrieve("Singapore impact on GlobalTech")
                    
                    # Should have both vector and graph context
                    assert result.vector_chunks is not None
                    assert result.graph_context is not None

    def test_should_handle_no_entities_found(self) -> None:
        """Should fallback to vector-only when no entities extracted."""
        with patch("src.retriever.hybrid.GraphDatabaseManager") as mock_graph:
            with patch("src.retriever.hybrid.VectorDatabaseManager") as mock_vector:
                with patch("src.retriever.hybrid.OllamaLLM") as mock_llm_class:
                    mock_graph_instance = MagicMock()
                    mock_graph.return_value = mock_graph_instance
                    
                    mock_vector_instance = MagicMock()
                    mock_vector.return_value = mock_vector_instance
                    mock_vector_instance.query_similar.return_value = [
                        {"id": "doc1", "document": "Some document", "distance": 0.1}
                    ]
                    
                    mock_llm = MagicMock()
                    mock_llm_class.return_value = mock_llm
                    mock_llm.invoke.return_value = '[]'
                    
                    from src.retriever.hybrid import HybridRetriever
                    
                    retriever = HybridRetriever()
                    result = retriever.retrieve("What is supply chain management?")
                    
                    # Should still return vector results
                    assert result.vector_chunks is not None


class TestRetrievalResult:
    """Tests for RetrievalResult data class."""

    def test_should_create_result(self) -> None:
        """Should create a retrieval result with context."""
        from src.retriever.hybrid import RetrievalResult
        
        result = RetrievalResult(
            query="test query",
            vector_chunks=["chunk1", "chunk2"],
            graph_context="path1 -> path2",
            entities=["entity1"],
            graph_paths=[]
        )
        
        assert result.query == "test query"
        assert len(result.vector_chunks) == 2

    def test_should_build_combined_context(self) -> None:
        """Should combine vector and graph context into single string."""
        from src.retriever.hybrid import RetrievalResult
        
        result = RetrievalResult(
            query="test query",
            vector_chunks=["Document about Singapore strike"],
            graph_context="Singapore -> FlowChips -> GlobalTech",
            entities=["Singapore"],
            graph_paths=[]
        )
        
        context = result.get_combined_context()
        
        assert "Singapore" in context
        assert "FlowChips" in context or "strike" in context
