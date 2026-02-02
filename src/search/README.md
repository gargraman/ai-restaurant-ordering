# Search System Documentation

## Overview
The search system implements hybrid retrieval combining BM25 (lexical), vector (semantic), and graph (relationship) search. It provides flexible search capabilities for the catering menu domain with multiple ranking strategies.

## Architecture
- **BM25 Search**: OpenSearch-based lexical search for exact matches
- **Vector Search**: pgvector-based semantic search for meaning matching
- **Graph Search**: Neo4j-based relationship search for connected data
- **Hybrid Search**: Combined search with RRF fusion for optimal results

## Key Components

### BM25 Search (`bm25.py`)
- OpenSearch lexical search with fuzzy matching
- Keyword-based search for exact matches
- Configurable scoring and ranking
- Advanced query parsing and analysis
- Performance optimization for large catalogs

### Vector Search (`vector.py`)
- pgvector semantic search via PostgreSQL
- Embedding-based similarity matching
- Cosine similarity calculations
- ANN (Approximate Nearest Neighbor) search
- Performance optimization for high-dimensional vectors

### Graph Search (`graph.py`)
- Neo4j graph search implementation with metrics
- Relationship-based search for connected concepts
- Cypher query optimization
- Graph traversal algorithms
- Performance monitoring for graph operations

### Hybrid Search (`hybrid.py`)
- Combined search with RRF (Reciprocal Rank Fusion) fusion
- Parallel execution of multiple search strategies
- Configurable weights for different search types
- Result merging and ranking
- Performance optimization for combined searches

## Key Features

### Hybrid Retrieval
- Simultaneous BM25 and vector search execution
- RRF fusion for optimal result combination
- Configurable weights for different search types
- Performance optimization for parallel execution
- Flexible result ranking strategies

### Semantic Understanding
- Vector embeddings for semantic matching
- Context-aware search results
- Meaning-based rather than keyword-based matching
- Handling of synonyms and related concepts
- Natural language query processing

### Lexical Matching
- Traditional keyword-based search
- Fuzzy matching for typos and variations
- Phrase and proximity search
- Field-specific search capabilities
- Performance optimization for large indexes

### Graph Relationships
- Relationship-based search for connected data
- Multi-hop traversal for complex queries
- Property graph querying
- Performance optimization for graph operations
- Integration with other search methods

### Performance Optimization
- Caching for frequent queries
- Index optimization for different search types
- Parallel processing for faster results
- Memory management for large datasets
- Query optimization for different search patterns

## Integration Points
- LangGraph pipeline for conversational search
- Data ingestion for index population
- API layer for search requests
- Model layer for query processing
- Monitoring for performance tracking

## Configuration
- Adjustable weights for different search types
- Configurable top-K values for each search
- RRF constant for rank fusion
- Performance parameters for each search type
- Index-specific settings for optimization

## Security
- Secure access to search infrastructure
- Input validation for search queries
- Protection against query injection
- Access controls for search results
- Encrypted communication with search services

## Error Handling
- Graceful degradation when search services are unavailable
- Fallback strategies for different search types
- Comprehensive error logging
- Circuit breaker patterns for service resilience
- Retry mechanisms for transient failures