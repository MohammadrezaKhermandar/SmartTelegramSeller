"""Custom exception types."""

from __future__ import annotations


class SalesAssistantError(Exception):
    """Base exception for the sales assistant."""


class ProductLoadError(SalesAssistantError):
    """Failed to load product data."""


class VectorStoreError(SalesAssistantError):
    """Vector store operation failed."""


class ToolExecutionError(SalesAssistantError):
    """A LangGraph tool failed."""


class LLMError(SalesAssistantError):
    """LLM call failed."""


class URLFetchError(SalesAssistantError):
    """URL fetching failed."""
