"""Common schemas: response wrappers, pagination, error models."""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """Standard API response wrapper."""

    data: T
    meta: dict[str, Any] | None = None


class PaginationMeta(BaseModel):
    """Pagination metadata."""

    total: int
    offset: int = 0
    limit: int = 50


class ErrorDetail(BaseModel):
    """Single error detail."""

    field: str | None = None
    message: str
    code: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    details: list[ErrorDetail] | None = None
    status_code: int = 500
