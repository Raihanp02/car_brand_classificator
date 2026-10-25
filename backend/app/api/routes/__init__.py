"""Route modules grouped by resource."""

from app.api.routes import admin, health, predict

ROUTERS = (health.router, predict.router, admin.router)

__all__ = ["ROUTERS", "admin", "health", "predict"]
