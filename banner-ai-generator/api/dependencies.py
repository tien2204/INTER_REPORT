from fastapi import HTTPException
from main import BannerGeneratorApp

# Global Banner app instance (được gán trong lifespan)
banner_app: BannerGeneratorApp = None

def set_banner_app(app: BannerGeneratorApp):
    """Set the global banner app instance (called on startup)"""
    global banner_app
    banner_app = app

def get_banner_app() -> BannerGeneratorApp:
    """Get the global banner app instance"""
    if not banner_app:
        raise HTTPException(status_code=503, detail="System not initialized")
    return banner_app
