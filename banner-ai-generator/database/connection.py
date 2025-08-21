"""
Database Connection Management

Handles SQLAlchemy engine creation, session management,
and connection pooling for both sync and async operations.
"""

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import asyncio
from typing import Optional, AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from structlog import get_logger

from config.system_config import get_system_config

logger = get_logger(__name__)


class DatabaseManager:
    """
    Database connection and session management
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_system_config().database
        
        # Connection settings
        self.database_url = self.config.get("url", "sqlite:///banner_generator.db")
        self.async_database_url = self.config.get("async_url", "sqlite+aiosqlite:///banner_generator.db")
        
        # Pool settings
        self.pool_size = self.config.get("pool_size", 10)
        self.max_overflow = self.config.get("max_overflow", 20)
        self.pool_timeout = self.config.get("pool_timeout", 30)
        self.pool_recycle = self.config.get("pool_recycle", 3600)
        
        # Initialize engines
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None
        
        self._initialized = False
    
    def initialize(self):
        """Initialize database engines and session factories"""
        try:
            logger.info("Initializing database connections")
            
            # Create synchronous engine
            engine_kwargs = {
                "echo": self.config.get("echo", False),
                "pool_pre_ping": True,
                "pool_recycle": self.pool_recycle
            }
            
            # Add pool settings for non-SQLite databases
            if not self.database_url.startswith("sqlite"):
                engine_kwargs.update({
                    "pool_size": self.pool_size,
                    "max_overflow": self.max_overflow,
                    "pool_timeout": self.pool_timeout
                })
            else:
                # SQLite-specific settings
                engine_kwargs.update({
                    "poolclass": StaticPool,
                    "connect_args": {"check_same_thread": False}
                })
            
            self.engine = create_engine(self.database_url, **engine_kwargs)
            
            # Create async engine
            async_engine_kwargs = {
                "echo": self.config.get("echo", False),
                "pool_pre_ping": True,
                "pool_recycle": self.pool_recycle
            }
            
            if not self.async_database_url.startswith("sqlite"):
                async_engine_kwargs.update({
                    "pool_size": self.pool_size,
                    "max_overflow": self.max_overflow,
                    "pool_timeout": self.pool_timeout
                })
            else:
                async_engine_kwargs.update({
                    "poolclass": StaticPool,
                    "connect_args": {"check_same_thread": False}
                })
            
            self.async_engine = create_async_engine(self.async_database_url, **async_engine_kwargs)
            
            # Create session factories
            self.session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            
            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            
            # Add connection event listeners
            self._setup_event_listeners()
            
            self._initialized = True
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _setup_event_listeners(self):
        """Setup database event listeners for monitoring and optimization"""
        
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for better performance"""
            if self.database_url.startswith("sqlite"):
                cursor = dbapi_connection.cursor()
                # Enable foreign key constraints
                cursor.execute("PRAGMA foreign_keys=ON")
                # Set journal mode to WAL for better concurrency
                cursor.execute("PRAGMA journal_mode=WAL")
                # Set synchronous mode to NORMAL for better performance
                cursor.execute("PRAGMA synchronous=NORMAL")
                # Set cache size (negative value means KB)
                cursor.execute("PRAGMA cache_size=-64000")  # 64MB
                cursor.close()
        
        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """Log connection checkout"""
            logger.debug("Database connection checked out")
        
        @event.listens_for(self.engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """Log connection checkin"""
            logger.debug("Database connection checked in")
    
    async def create_tables(self):
        """Create all database tables"""
        try:
            if not self._initialized:
                self.initialize()
            
            logger.info("Creating database tables")
            
            from .models import Base
            
            # Create tables using async engine
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    async def drop_tables(self):
        """Drop all database tables"""
        try:
            if not self._initialized:
                self.initialize()
            
            logger.warning("Dropping all database tables")
            
            from .models import Base
            
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            
            logger.warning("All database tables dropped")
            
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new synchronous database session"""
        if not self._initialized:
            self.initialize()
        
        return self.session_factory()
    
    def get_async_session(self) -> AsyncSession:
        """Get a new asynchronous database session"""
        if not self._initialized:
            self.initialize()
        
        return self.async_session_factory()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def async_session_scope(self) -> AsyncGenerator[AsyncSession, None]:
        """Provide an async transactional scope around a series of operations"""
        session = self.get_async_session()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Async database session error: {e}")
            raise
        finally:
            await session.close()
    
    async def health_check(self) -> dict:
        """Perform database health check"""
        try:
            if not self._initialized:
                self.initialize()
            
            # Test sync connection
            sync_healthy = False
            try:
                with self.session_scope() as session:
                    result = session.execute(text("SELECT 1"))
                    sync_healthy = result.scalar() == 1
            except Exception as e:
                logger.error(f"Sync database health check failed: {e}")
            
            # Test async connection
            async_healthy = False
            try:
                async with self.async_session_scope() as session:
                    result = await session.execute(text("SELECT 1"))
                    async_healthy = result.scalar() == 1
            except Exception as e:
                logger.error(f"Async database health check failed: {e}")
            
            return {
                "sync_connection": "healthy" if sync_healthy else "unhealthy",
                "async_connection": "healthy" if async_healthy else "unhealthy",
                "overall": "healthy" if sync_healthy and async_healthy else "unhealthy",
                "database_url": self.database_url.split("@")[-1] if "@" in self.database_url else self.database_url,
                "pool_size": self.pool_size,
                "max_overflow": self.max_overflow
            }
            
        except Exception as e:
            logger.error(f"Database health check error: {e}")
            return {
                "overall": "unhealthy",
                "error": str(e)
            }
    
    async def get_connection_stats(self) -> dict:
        """Get database connection pool statistics"""
        try:
            if not self._initialized:
                return {"error": "Database not initialized"}
            
            pool = self.engine.pool
            
            return {
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "total_connections": pool.size() + pool.overflow()
            }
            
        except Exception as e:
            logger.error(f"Error getting connection stats: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close all database connections"""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("Sync database engine disposed")
            
            if self.async_engine:
                await self.async_engine.dispose()
                logger.info("Async database engine disposed")
            
            self._initialized = False
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.initialize()
    return _db_manager


def get_db_session() -> Session:
    """Get a synchronous database session"""
    return get_database_manager().get_session()


def get_async_db_session() -> AsyncSession:
    """Get an asynchronous database session"""
    return get_database_manager().get_async_session()


# Dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for getting database session"""
    db_manager = get_database_manager()
    async with db_manager.async_session_scope() as session:
        yield session
