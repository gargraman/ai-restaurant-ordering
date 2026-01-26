-- Initialize pgvector extension and create schema

CREATE EXTENSION IF NOT EXISTS vector;

-- Menu embeddings table for vector search
CREATE TABLE IF NOT EXISTS menu_embeddings (
    doc_id UUID PRIMARY KEY,
    embedding vector(1536),
    restaurant_id VARCHAR(64),
    city VARCHAR(100),
    base_price DECIMAL(10,2),
    serves_max INTEGER,
    dietary_labels TEXT[],
    created_at TIMESTAMP DEFAULT NOW()
);

-- Vector similarity search index (IVFFlat)
CREATE INDEX IF NOT EXISTS idx_menu_embeddings_vector
    ON menu_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Supporting indexes for filtered search
CREATE INDEX IF NOT EXISTS idx_menu_embeddings_city
    ON menu_embeddings (city);

CREATE INDEX IF NOT EXISTS idx_menu_embeddings_price
    ON menu_embeddings (base_price);

CREATE INDEX IF NOT EXISTS idx_menu_embeddings_serves
    ON menu_embeddings (serves_max);

CREATE INDEX IF NOT EXISTS idx_menu_embeddings_restaurant
    ON menu_embeddings (restaurant_id);

-- Function for filtered vector search
CREATE OR REPLACE FUNCTION search_menu_embeddings(
    query_embedding vector(1536),
    filter_city VARCHAR DEFAULT NULL,
    filter_price_max DECIMAL DEFAULT NULL,
    filter_serves_min INTEGER DEFAULT NULL,
    filter_dietary TEXT[] DEFAULT NULL,
    result_limit INTEGER DEFAULT 50
)
RETURNS TABLE (
    doc_id UUID,
    score FLOAT,
    restaurant_id VARCHAR,
    city VARCHAR,
    base_price DECIMAL,
    serves_max INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        me.doc_id,
        1 - (me.embedding <=> query_embedding) as score,
        me.restaurant_id,
        me.city,
        me.base_price,
        me.serves_max
    FROM menu_embeddings me
    WHERE
        (filter_city IS NULL OR me.city = filter_city)
        AND (filter_price_max IS NULL OR me.base_price <= filter_price_max)
        AND (filter_serves_min IS NULL OR me.serves_max >= filter_serves_min)
        AND (filter_dietary IS NULL OR me.dietary_labels && filter_dietary)
    ORDER BY me.embedding <=> query_embedding
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO postgres;
