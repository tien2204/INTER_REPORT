"""
API Routers

This module contains all the FastAPI routers for different
API endpoint groups.
"""

from . import campaigns
from . import assets
from . import designs
from . import agents

__all__ = [
    "campaigns",
    "assets", 
    "designs",
    "agents"
]
