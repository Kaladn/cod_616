"""Integration helper to wire ProcessService into an EventManager.

Testable in isolation (no heavy harness imports required).
"""
import logging
from typing import Optional

try:
    from .process_service import ProcessService
except Exception:
    raise


def integrate_process_logger(event_mgr, config: dict, env: Optional[dict] = None):
    """Start ProcessService and register its EventManager source.

    Args:
        event_mgr: EventManager-like object with `register_source` method
        config: dict with optional key `loggers.process` child with keys:
            - enabled (bool)
            - config_path (str)
            - use_subprocess (bool)
        env: optional dict of environment variables (defaults to os.environ)

    Returns:
        ProcessService instance if started, otherwise None
    """
    env = env or {}
    process_cfg = config.get('loggers', {}).get('process', {})
    process_enabled = process_cfg.get('enabled', False) or env.get('ENABLE_PROCESS_LOGGER') == '1'

    if not process_enabled:
        logging.info("Process logging not enabled")
        return None

    try:
        service = ProcessService(config_path=process_cfg.get('config_path'), use_subprocess=process_cfg.get('use_subprocess', True))
        service.start()
        event_mgr.register_source('process', 'logger', {'type': 'ProcessLogger'})
        logging.info("ProcessLogger started and registered as 'process'")
        return service
    except Exception as e:
        logging.warning(f"Failed to start ProcessLogger: {e}")
        return None