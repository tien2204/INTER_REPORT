"""
Persistent memory storage for campaigns, designs, and feedback
"""

import json
import sqlite3
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class DesignVersion:
    """Design version data structure"""
    version_id: str
    session_id: str
    blueprint: Dict[str, Any]
    background_url: Optional[str]
    created_at: datetime
    created_by: str
    feedback: Optional[Dict[str, Any]] = None
    status: str = "active"  # active, archived, deleted

@dataclass
class CampaignData:
    """Campaign data structure"""
    campaign_id: str
    name: str
    brief: Dict[str, Any]
    brand_assets: Dict[str, Any]
    target_audience: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    status: str = "active"
    metadata: Optional[Dict[str, Any]] = None

class MemoryStore:
    """
    In-memory storage with optional persistence
    Fast access for current session data
    """
    
    def __init__(self):
        self._campaigns: Dict[str, CampaignData] = {}
        self._designs: Dict[str, List[DesignVersion]] = {}
        self._feedback: Dict[str, List[Dict[str, Any]]] = {}
        self._assets: Dict[str, Dict[str, Any]] = {}
    
    def store_campaign(self, campaign: CampaignData) -> None:
        """Store campaign data"""
        self._campaigns[campaign.campaign_id] = campaign
        logger.debug(f"Stored campaign: {campaign.campaign_id}")
    
    def get_campaign(self, campaign_id: str) -> Optional[CampaignData]:
        """Retrieve campaign data"""
        return self._campaigns.get(campaign_id)
    
    def list_campaigns(self) -> List[CampaignData]:
        """List all campaigns"""
        return list(self._campaigns.values())
    
    def store_design_version(self, session_id: str, design: DesignVersion) -> None:
        """Store design version"""
        if session_id not in self._designs:
            self._designs[session_id] = []
        self._designs[session_id].append(design)
        logger.debug(f"Stored design version: {design.version_id}")
    
    def get_design_versions(self, session_id: str) -> List[DesignVersion]:
        """Get all design versions for a session"""
        return self._designs.get(session_id, [])
    
    def get_latest_design(self, session_id: str) -> Optional[DesignVersion]:
        """Get latest design version"""
        versions = self._designs.get(session_id, [])
        return versions[-1] if versions else None
    
    def store_feedback(self, session_id: str, feedback: Dict[str, Any]) -> None:
        """Store feedback"""
        if session_id not in self._feedback:
            self._feedback[session_id] = []
        feedback['timestamp'] = datetime.now().isoformat()
        self._feedback[session_id].append(feedback)
        logger.debug(f"Stored feedback for session: {session_id}")
    
    def get_feedback(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all feedback for a session"""
        return self._feedback.get(session_id, [])
    
    def store_asset(self, asset_id: str, asset_data: Dict[str, Any]) -> None:
        """Store asset data"""
        self._assets[asset_id] = asset_data
    
    def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Get asset data"""
        return self._assets.get(asset_id)
    
    def clear(self) -> None:
        """Clear all in-memory data"""
        self._campaigns.clear()
        self._designs.clear()
        self._feedback.clear()
        self._assets.clear()
        logger.info("Cleared all memory store data")

class PersistentStore:
    """
    Persistent storage using SQLite for data durability
    Handles long-term storage and retrieval
    """
    
    def __init__(self, db_path: Union[str, Path] = "banner_ai_storage.db"):
        self.db_path = Path(db_path)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database tables"""
        with self.get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    brief TEXT NOT NULL,
                    brand_assets TEXT NOT NULL,
                    target_audience TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    status TEXT DEFAULT 'active',
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS design_versions (
                    version_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    campaign_id TEXT,
                    blueprint TEXT NOT NULL,
                    background_url TEXT,
                    created_at TIMESTAMP NOT NULL,
                    created_by TEXT NOT NULL,
                    feedback TEXT,
                    status TEXT DEFAULT 'active',
                    FOREIGN KEY (campaign_id) REFERENCES campaigns (campaign_id)
                );
                
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    version_id TEXT,
                    feedback_data TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    feedback_type TEXT DEFAULT 'user',
                    FOREIGN KEY (version_id) REFERENCES design_versions (version_id)
                );
                
                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY,
                    campaign_id TEXT,
                    asset_type TEXT NOT NULL,
                    file_path TEXT,
                    metadata TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (campaign_id) REFERENCES campaigns (campaign_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_design_session ON design_versions(session_id);
                CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id);
                CREATE INDEX IF NOT EXISTS idx_assets_campaign ON assets(campaign_id);
            """)
        logger.info(f"Initialized database: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def save_campaign(self, campaign: CampaignData) -> None:
        """Save campaign to persistent storage"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO campaigns 
                (campaign_id, name, brief, brand_assets, target_audience, 
                 created_at, updated_at, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                campaign.campaign_id,
                campaign.name,
                json.dumps(campaign.brief),
                json.dumps(campaign.brand_assets),
                json.dumps(campaign.target_audience),
                campaign.created_at,
                campaign.updated_at,
                campaign.status,
                json.dumps(campaign.metadata) if campaign.metadata else None
            ))
            conn.commit()
        logger.debug(f"Saved campaign to persistent storage: {campaign.campaign_id}")
    
    def load_campaign(self, campaign_id: str) -> Optional[CampaignData]:
        """Load campaign from persistent storage"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM campaigns WHERE campaign_id = ?",
                (campaign_id,)
            ).fetchone()
            
            if not row:
                return None
            
            return CampaignData(
                campaign_id=row['campaign_id'],
                name=row['name'],
                brief=json.loads(row['brief']),
                brand_assets=json.loads(row['brand_assets']),
                target_audience=json.loads(row['target_audience']),
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                status=row['status'],
                metadata=json.loads(row['metadata']) if row['metadata'] else None
            )
    
    def save_design_version(self, design: DesignVersion, campaign_id: Optional[str] = None) -> None:
        """Save design version to persistent storage"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO design_versions 
                (version_id, session_id, campaign_id, blueprint, background_url,
                 created_at, created_by, feedback, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                design.version_id,
                design.session_id,
                campaign_id,
                json.dumps(design.blueprint),
                design.background_url,
                design.created_at,
                design.created_by,
                json.dumps(design.feedback) if design.feedback else None,
                design.status
            ))
            conn.commit()
        logger.debug(f"Saved design version: {design.version_id}")
    
    def load_design_versions(self, session_id: str) -> List[DesignVersion]:
        """Load all design versions for a session"""
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM design_versions 
                WHERE session_id = ? 
                ORDER BY created_at ASC
            """, (session_id,)).fetchall()
            
            versions = []
            for row in rows:
                versions.append(DesignVersion(
                    version_id=row['version_id'],
                    session_id=row['session_id'],
                    blueprint=json.loads(row['blueprint']),
                    background_url=row['background_url'],
                    created_at=row['created_at'],
                    created_by=row['created_by'],
                    feedback=json.loads(row['feedback']) if row['feedback'] else None,
                    status=row['status']
                ))
            
            return versions
    
    def save_feedback(self, session_id: str, feedback: Dict[str, Any], 
                      version_id: Optional[str] = None, feedback_type: str = "user") -> None:
        """Save feedback to persistent storage"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO feedback (session_id, version_id, feedback_data, created_at, feedback_type)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                version_id,
                json.dumps(feedback),
                datetime.now(),
                feedback_type
            ))
            conn.commit()
        logger.debug(f"Saved feedback for session: {session_id}")
    
    def load_feedback(self, session_id: str) -> List[Dict[str, Any]]:
        """Load all feedback for a session"""
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM feedback 
                WHERE session_id = ? 
                ORDER BY created_at ASC
            """, (session_id,)).fetchall()
            
            feedback_list = []
            for row in rows:
                feedback_data = json.loads(row['feedback_data'])
                feedback_data['id'] = row['id']
                feedback_data['version_id'] = row['version_id']
                feedback_data['created_at'] = row['created_at']
                feedback_data['feedback_type'] = row['feedback_type']
                feedback_list.append(feedback_data)
            
            return feedback_list
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """Cleanup data older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self.get_connection() as conn:
            # Delete old feedback
            cursor = conn.execute("""
                DELETE FROM feedback WHERE created_at < ?
            """, (cutoff_date,))
            deleted_feedback = cursor.rowcount
            
            # Delete old design versions
            cursor = conn.execute("""
                DELETE FROM design_versions WHERE created_at < ? AND status != 'archived'
            """, (cutoff_date,))
            deleted_designs = cursor.rowcount
            
            conn.commit()
            
        total_deleted = deleted_feedback + deleted_designs
        logger.info(f"Cleaned up {total_deleted} old records")
        return total_deleted
