"""
Database Module for Banner AI Generator

This module provides database models, connection management,
and data access layers using SQLAlchemy ORM.
"""

from .connection import DatabaseManager, get_db_session, get_async_db_session
from .models import (
    Base,
    Campaign, 
    Asset,
    Design,
    User,
    Session,
    GenerationMetrics,
    Feedback
)
from .repositories import (
    CampaignRepository,
    AssetRepository, 
    DesignRepository,
    UserRepository,
    MetricsRepository
)

__all__ = [
    "DatabaseManager",
    "get_db_session",
    "get_async_db_session",
    "Base",
    "Campaign",
    "Asset", 
    "Design",
    "User",
    "Session",
    "GenerationMetrics",
    "Feedback",
    "CampaignRepository",
    "AssetRepository",
    "DesignRepository", 
    "UserRepository",
    "MetricsRepository"
]
