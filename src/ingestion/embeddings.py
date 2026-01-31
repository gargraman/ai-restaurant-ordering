"""Embedding generation for vector search."""

import time
from typing import Sequence

import structlog
from openai import AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config import get_settings
from src.models.index import IndexDocument
from src.metrics import record_llm_call

logger = structlog.get_logger()

# OpenAI embedding API limits
MAX_BATCH_SIZE = 2048  # OpenAI's maximum inputs per request


class EmbeddingGenerator:
    """Generate embeddings for documents using OpenAI."""

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        batch_size: int = 1000,
        fail_on_batch_error: bool = True,
    ):
        """Initialize embedding generator.

        Args:
            client: Optional AsyncOpenAI client
            batch_size: Number of texts per batch (max 2048, default 1000 for safety)
            fail_on_batch_error: If True, raise on batch error; if False, skip failed batches
        """
        settings = get_settings()
        self.client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model
        self.dimensions = settings.embedding_dimensions
        self._batch_size = min(batch_size, MAX_BATCH_SIZE)
        self._fail_on_batch_error = fail_on_batch_error

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((RateLimitError, TimeoutError)),
    )
    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        start_time = time.time()
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
            )
            duration = time.time() - start_time

            # Extract token usage if available
            usage = getattr(response, 'usage', None)
            input_tokens = usage.prompt_tokens if usage else 0

            # Record LLM metrics
            record_llm_call(
                model=self.model,
                operation='embedding_generation',
                duration=duration,
                input_tokens=input_tokens
            )

            return response.data[0].embedding
        except Exception as e:
            duration = time.time() - start_time
            record_llm_call(
                model=self.model,
                operation='embedding_generation',
                duration=duration
            )
            raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((RateLimitError, TimeoutError)),
    )
    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed (max 2048)

        Returns:
            List of embeddings in same order as input texts

        Raises:
            ValueError: If batch size exceeds OpenAI limit
        """
        if not texts:
            return []

        if len(texts) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(texts)} exceeds OpenAI limit of {MAX_BATCH_SIZE}"
            )

        start_time = time.time()
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
            )
            duration = time.time() - start_time

            # Extract token usage if available
            usage = getattr(response, 'usage', None)
            input_tokens = usage.prompt_tokens if usage else 0

            # Record LLM metrics
            record_llm_call(
                model=self.model,
                operation='embedding_batch_generation',
                duration=duration,
                input_tokens=input_tokens
            )

            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
        except Exception as e:
            duration = time.time() - start_time
            record_llm_call(
                model=self.model,
                operation='embedding_batch_generation',
                duration=duration
            )
            raise

    async def generate_document_embeddings(
        self,
        documents: Sequence[IndexDocument],
    ) -> dict[str, list[float]]:
        """Generate embeddings for all documents.

        Args:
            documents: Sequence of IndexDocuments to embed

        Returns:
            Dictionary mapping doc_id to embedding vector
        """
        if not documents:
            logger.warning("no_documents_for_embedding")
            return {}

        logger.info("generating_embeddings", document_count=len(documents))

        # Prepare texts for embedding
        doc_texts = [(doc.doc_id, doc.text) for doc in documents]

        embeddings: dict[str, list[float]] = {}
        failed_batches: list[int] = []
        total_batches = (len(doc_texts) + self._batch_size - 1) // self._batch_size

        for batch_num in range(total_batches):
            start_idx = batch_num * self._batch_size
            end_idx = min(start_idx + self._batch_size, len(doc_texts))
            batch = doc_texts[start_idx:end_idx]

            logger.info(
                "processing_embedding_batch",
                batch=batch_num + 1,
                total_batches=total_batches,
                batch_size=len(batch),
            )

            texts = [text for _, text in batch]
            doc_ids = [doc_id for doc_id, _ in batch]

            try:
                batch_embeddings = await self.generate_embeddings_batch(texts)

                for doc_id, embedding in zip(doc_ids, batch_embeddings):
                    embeddings[doc_id] = embedding

            except Exception as e:
                logger.error(
                    "batch_embedding_error",
                    batch=batch_num + 1,
                    total_batches=total_batches,
                    batch_size=len(batch),
                    error=str(e),
                    error_type=type(e).__name__,
                )

                if self._fail_on_batch_error:
                    raise

                # Graceful degradation: skip failed batch and continue
                failed_batches.append(batch_num + 1)
                logger.warning(
                    "batch_skipped",
                    batch=batch_num + 1,
                    skipped_doc_ids=doc_ids,
                    skipped_count=len(doc_ids),
                )

        if failed_batches:
            logger.warning(
                "embedding_generation_partial",
                total_embeddings=len(embeddings),
                failed_batches=failed_batches,
                failed_batch_count=len(failed_batches),
            )
        else:
            logger.info("embeddings_complete", total_embeddings=len(embeddings))

        return embeddings

    async def generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for a search query."""
        return await self.generate_embedding(query)


def get_embedding_generator(
    batch_size: int = 1000,
    fail_on_batch_error: bool = True,
) -> EmbeddingGenerator:
    """Get embedding generator instance.

    Args:
        batch_size: Number of texts per batch (default 1000)
        fail_on_batch_error: If True, raise on batch error; if False, skip failed batches
    """
    return EmbeddingGenerator(
        batch_size=batch_size,
        fail_on_batch_error=fail_on_batch_error,
    )
