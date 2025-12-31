"""Integration helper to wire ActivityService into an EventManager.

This helper can be unit-tested in isolation without importing the full
`truevision_event_live` harness, avoiding heavy third-party imports.
"""
import os
import logging
from typing import Optional

try:
    from .activity_service import ActivityService
except Exception:
    # In test environments this may not be importable; re-raise with context
    raise


def integrate_activity_logger(event_mgr, config: dict, env: Optional[dict] = None):
    """Start ActivityService and register its EventManager source.

    Args:
        event_mgr: EventManager-like object with `register_source` method
        config: dict with optional key `loggers.activity` child with keys:
            - enabled (bool)
            - config_path (str)
            - use_subprocess (bool)
        env: optional dict of environment variables (defaults to os.environ)

    Returns:
        ActivityService instance if started, otherwise None
    """
    env = env or os.environ
    activity_cfg = config.get('loggers', {}).get('activity', {})
    activity_enabled = activity_cfg.get('enabled', False) or env.get('ENABLE_ACTIVITY_LOGGER') == '1'

    if not activity_enabled:
        logging.info("Activity logging not enabled")
        return None

    try:
        service = ActivityService(config_path=activity_cfg.get('config_path'), use_subprocess=activity_cfg.get('use_subprocess', True))
        service.start()
        event_mgr.register_source('activity', 'logger', {'type': 'ActivityLogger'})
        logging.info("ActivityLogger started and registered as 'activity'")
        return service
    except Exception as e:
        logging.warning(f"Failed to start ActivityLogger: {e}")
        return None
