"""
Hybrid Retriever for Atlas-GRAG.

Combines vector-based semantic search with graph-based multi-hop
reasoning to provide context for answering complex supply chain questions.

This is the "brain" of Atlas-GRAG - it enables answering questions that
require connecting multiple pieces of information through graph traversal.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_ollama import OllamaLLM

from src.config import get_config
from src.database.graph_db import GraphDatabaseManager
from src.database.vector_db import VectorDatabaseManager

logger = logging.getLogger(__name__)


# Prompt for entity extraction from queries
ENTITY_EXTRACTION_PROMPT = """You are an entity extraction system for supply chain analysis.
Extract all named entities (companies, products, locations, events) from the user's question.

Return ONLY a JSON array of entity names, nothing else.

Example:
Question: "How will the Singapore port strike affect GlobalTech's production?"
Answer: ["Singapore", "GlobalTech"]

Question: "{query}"
Answer:"""


@dataclass
class GraphPath:
    """
    Represents a path through the knowledge graph.
    
    Attributes:
        nodes: List of node names in the path
        relationships: List of relationship types
        path_length: Number of hops
    """
    nodes: List[str]
    relationships: List[str]
    path_length: int
    
    def to_string(self) -> str:
        """Convert path to human-readable string."""
        if not self.nodes:
            return ""
        
        parts = []
        for i, node in enumerate(self.nodes):
            parts.append(node)
            if i < len(self.relationships):
                parts.append(f"-[{self.relationships[i]}]->")
        
        return " ".join(parts)


@dataclass
class RetrievalResult:
    """
    Result of hybrid retrieval containing both vector and graph context.
    
    Attributes:
        query: Original user query
        vector_chunks: List of similar document chunks
        graph_context: Formatted graph relationships
        entities: Entities extracted from query
        graph_paths: Paths found in the knowledge graph
    """
    query: str
    vector_chunks: List[str] = field(default_factory=list)
    graph_context: str = ""
    entities: List[str] = field(default_factory=list)
    graph_paths: List[GraphPath] = field(default_factory=list)
    
    def get_combined_context(self) -> str:
        """
        Combine vector and graph context into a single context string.
        
        Returns:
            Combined context for LLM consumption
        """
        sections = []
        
        if self.vector_chunks:
            sections.append("## Relevant Documents")
            for i, chunk in enumerate(self.vector_chunks, 1):
                sections.append(f"{i}. {chunk}")
        
        if self.graph_context:
            sections.append("\n## Knowledge Graph Relationships")
            sections.append(self.graph_context)
        
        if self.graph_paths:
            sections.append("\n## Graph Paths")
            for path in self.graph_paths:
                sections.append(f"- {path.to_string()}")
        
        return "\n".join(sections)


class HybridRetriever:
    """
    Hybrid retriever combining vector search and graph traversal.
    
    The retrieval process:
    1. Extract entities from the user query using LLM
    2. Search ChromaDB for semantically similar documents
    3. For each entity, find N-hop neighbors in Neo4j
    4. Combine results into unified context
    
    Example:
        retriever = HybridRetriever()
        result = retriever.retrieve(
            "How will the Singapore strike impact GlobalTech?"
        )
        print(result.get_combined_context())
    """
    
    def __init__(
        self,
        graph_manager: Optional[GraphDatabaseManager] = None,
        vector_manager: Optional[VectorDatabaseManager] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Initialize the hybrid retriever.
        
        Args:
            graph_manager: Neo4j connection (created if not provided)
            vector_manager: ChromaDB manager (created if not provided)
            model: Ollama model for entity extraction
        """
        config = get_config()
        
        self._graph = graph_manager or GraphDatabaseManager()
        self._vector = vector_manager or VectorDatabaseManager()
        self._model = model or config.ollama.model
        
        self._llm = OllamaLLM(
            model=self._model,
            base_url=config.ollama.base_url,
            temperature=0.0
        )
        
        self._vector_top_k = config.retrieval.vector_top_k
        self._max_hops = config.retrieval.graph_max_hops
        self._collection_name = config.chroma.collection_name
    
    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract entity names from user query using LLM.
        
        Args:
            query: User's question
            
        Returns:
            List of entity names
        """
        prompt = ENTITY_EXTRACTION_PROMPT.format(query=query)
        
        try:
            response = self._llm.invoke(prompt)
            
            # Parse JSON array from response
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                entities = json.loads(json_match.group())
                if isinstance(entities, list):
                    return [str(e) for e in entities]
            
            logger.warning(f"Could not parse entities from: {response}")
            return []
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    def _retrieve_vector(
        self,
        query: str,
        n_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents from vector store.
        
        Args:
            query: Search query
            n_results: Number of results (defaults to config)
            
        Returns:
            List of similar documents
        """
        try:
            return self._vector.query_similar(
                collection_name=self._collection_name,
                query_text=query,
                n_results=n_results or self._vector_top_k
            )
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return []
    
    def _build_neighbor_query(self, entity: str, max_hops: int = 2) -> str:
        """
        Build Cypher query to find neighbors of an entity.
        
        Args:
            entity: Entity name to search from
            max_hops: Maximum path length
            
        Returns:
            Cypher query string
        """
        return f"""
        MATCH (n)
        WHERE n.name =~ '(?i).*{entity}.*'
        MATCH path = (n)-[*1..{max_hops}]-(neighbor)
        WHERE neighbor <> n
        RETURN DISTINCT n.name AS source, 
               neighbor.name AS target,
               [r IN relationships(path) | type(r)] AS relationships,
               length(path) AS path_length
        ORDER BY path_length
        LIMIT 10
        """
    
    def _retrieve_graph_neighbors(
        self,
        entity: str,
        max_hops: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find entities within N hops of the given entity.
        
        Args:
            entity: Entity name to start from
            max_hops: Maximum hops (defaults to config)
            
        Returns:
            List of neighbor information
        """
        try:
            # First try to find the exact entity
            query = self._build_neighbor_query(entity, max_hops or self._max_hops)
            return self._graph.execute_query(query)
        except Exception as e:
            logger.error(f"Graph neighbor retrieval failed: {e}")
            return []
    
    def _get_paths_between_entities(
        self,
        entities: List[str]
    ) -> List[GraphPath]:
        """
        Find paths connecting multiple entities.
        
        Args:
            entities: List of entity names
            
        Returns:
            List of paths between entities
        """
        paths = []
        
        if len(entities) < 2:
            return paths
        
        try:
            # Try to find paths between first two entities
            query = f"""
            MATCH (a), (b)
            WHERE a.name =~ '(?i).*{entities[0]}.*'
              AND b.name =~ '(?i).*{entities[1]}.*'
            MATCH path = shortestPath((a)-[*..{self._max_hops + 1}]-(b))
            RETURN [n IN nodes(path) | n.name] AS nodes,
                   [r IN relationships(path) | type(r)] AS relationships,
                   length(path) AS path_length
            LIMIT 3
            """
            
            results = self._graph.execute_query(query)
            
            for result in results:
                paths.append(GraphPath(
                    nodes=result.get("nodes", []),
                    relationships=result.get("relationships", []),
                    path_length=result.get("path_length", 0)
                ))
                
        except Exception as e:
            logger.warning(f"Path finding failed: {e}")
        
        return paths
    
    def _format_graph_context(
        self,
        neighbors: List[Dict[str, Any]],
        paths: List[GraphPath]
    ) -> str:
        """
        Format graph results into readable context.
        
        Args:
            neighbors: Neighbor query results
            paths: Paths between entities
            
        Returns:
            Formatted context string
        """
        lines = []
        
        # Format neighbor relationships
        seen = set()
        for neighbor in neighbors:
            source = neighbor.get("source", "")
            target = neighbor.get("target", "")
            rels = neighbor.get("relationships", [])
            
            if source and target:
                key = f"{source}-{target}"
                if key not in seen:
                    seen.add(key)
                    rel_str = " -> ".join(rels) if rels else "RELATED"
                    lines.append(f"- {source} --[{rel_str}]--> {target}")
        
        # Format paths
        for path in paths:
            lines.append(f"- Path: {path.to_string()}")
        
        return "\n".join(lines)
    
    def retrieve(
        self,
        query: str,
        include_graph: bool = True
    ) -> RetrievalResult:
        """
        Perform hybrid retrieval combining vector and graph search.
        
        Args:
            query: User's question
            include_graph: Whether to include graph traversal
            
        Returns:
            RetrievalResult with combined context
        """
        result = RetrievalResult(query=query)
        
        # Step 1: Extract entities from query
        if include_graph:
            entities = self._extract_entities(query)
            result.entities = entities
            logger.info(f"Extracted entities: {entities}")
        else:
            entities = []
        
        # Step 2: Vector retrieval
        vector_docs = self._retrieve_vector(query)
        result.vector_chunks = [
            doc.get("document", "") 
            for doc in vector_docs 
            if doc.get("document")
        ]
        
        # Step 3: Graph retrieval for each entity
        if entities:
            all_neighbors = []
            for entity in entities:
                neighbors = self._retrieve_graph_neighbors(entity)
                all_neighbors.extend(neighbors)
            
            # Step 4: Find paths between entities
            paths = self._get_paths_between_entities(entities)
            result.graph_paths = paths
            
            # Step 5: Format graph context
            result.graph_context = self._format_graph_context(all_neighbors, paths)
        
        return result
    
    def retrieve_with_fallback(
        self,
        query: str
    ) -> RetrievalResult:
        """
        Retrieve with graceful fallback if graph database is unavailable.
        
        Args:
            query: User's question
            
        Returns:
            RetrievalResult (vector-only if graph fails)
        """
        # Check if graph is available
        try:
            if not self._graph.is_healthy():
                logger.warning("Graph database unavailable, using vector-only retrieval")
                return self.retrieve(query, include_graph=False)
        except Exception:
            logger.warning("Graph health check failed, using vector-only retrieval")
            return self.retrieve(query, include_graph=False)
        
        return self.retrieve(query, include_graph=True)
