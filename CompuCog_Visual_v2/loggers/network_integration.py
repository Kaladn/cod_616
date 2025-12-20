"""Integration helper to wire NetworkService into an EventManager.

Testable in isolation.
"""
import logging
from typing import Optional

try:
    from .network_service import NetworkService
except Exception:
    raise


def integrate_network_logger(event_mgr, config: dict, env: Optional[dict] = None):
    """Start NetworkService and register its EventManager source.

    Args:
        event_mgr: EventManager-like object with `register_source` method
        config: dict with optional key `loggers.network` child with keys:
            - enabled (bool)
            - config_path (str)
            - use_subprocess (bool)
        env: optional dict of environment variables (defaults to os.environ)

    Returns:
        NetworkService instance if started, otherwise None
    """
    env = env or {}
    net_cfg = config.get('loggers', {}).get('network', {})
    net_enabled = net_cfg.get('enabled', False) or env.get('ENABLE_NETWORK_LOGGER') == '1'

    if not net_enabled:
        logging.info("Network logging not enabled")
        return None

    try:
        service = NetworkService(config_path=net_cfg.get('config_path'), use_subprocess=net_cfg.get('use_subprocess', True))
        service.start()
        event_mgr.register_source('network', 'logger', {'type': 'NetworkLogger'})
        logging.info("NetworkLogger started and registered as 'network'")
        return service
    except Exception as e:
        logging.warning(f"Failed to start NetworkLogger: {e}")
        return None