"""Base data models for the trading system."""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


class BaseEntity(BaseModel):
    """Base entity with common fields."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()


class TimeSeriesEntity(BaseEntity):
    """Base entity with timestamp for time series data."""
    
    timestamp: datetime = Field(..., description="Data timestamp")
    symbol: str = Field(..., description="Trading symbol")
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ConfigurableEntity(BaseEntity):
    """Base entity with configuration support."""
    
    name: str = Field(..., description="Entity name")
    version: str = Field(..., description="Entity version")
    description: Optional[str] = Field(None, description="Entity description")
    enabled: bool = Field(default=True, description="Whether entity is enabled")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration parameters")
