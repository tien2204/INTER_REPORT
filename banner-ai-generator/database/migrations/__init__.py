"""
Database Migrations

This module handles database schema migrations using Alembic.
"""

from alembic.config import Config
from alembic import command
import os
from structlog import get_logger

logger = get_logger(__name__)


def get_alembic_config():
    """Get Alembic configuration"""
    # Get the directory containing this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to the project root
    project_root = os.path.dirname(os.path.dirname(current_dir))
    alembic_cfg_path = os.path.join(project_root, "alembic.ini")
    
    config = Config(alembic_cfg_path)
    
    # Set the script location to the migrations directory
    config.set_main_option("script_location", os.path.join(current_dir))
    
    return config


def create_migration(message: str):
    """Create a new migration"""
    try:
        config = get_alembic_config()
        command.revision(config, message=message, autogenerate=True)
        logger.info(f"Migration created: {message}")
    except Exception as e:
        logger.error(f"Error creating migration: {e}")
        raise


def upgrade_database(revision: str = "head"):
    """Upgrade database to specified revision"""
    try:
        config = get_alembic_config()
        command.upgrade(config, revision)
        logger.info(f"Database upgraded to {revision}")
    except Exception as e:
        logger.error(f"Error upgrading database: {e}")
        raise


def downgrade_database(revision: str):
    """Downgrade database to specified revision"""
    try:
        config = get_alembic_config()
        command.downgrade(config, revision)
        logger.info(f"Database downgraded to {revision}")
    except Exception as e:
        logger.error(f"Error downgrading database: {e}")
        raise


def get_migration_history():
    """Get migration history"""
    try:
        config = get_alembic_config()
        # This would return migration history
        # Implementation depends on specific requirements
        return []
    except Exception as e:
        logger.error(f"Error getting migration history: {e}")
        return []
