"""Integration helper to wire InputService into an EventManager.

Testable in isolation (no heavy harness imports required).
"""
import logging
from typing import Optional

try:
    from .input_service import InputService
except Exception:
    raise


def integrate_input_logger(event_mgr, config: dict, env: Optional[dict] = None):
    """Start InputService and register its EventManager source.

    Args:
        event_mgr: EventManager-like object with `register_source` method
        config: dict with optional key `loggers.input` child with keys:
            - enabled (bool)
            - config_path (str)
            - use_subprocess (bool)
        env: optional dict of environment variables (defaults to os.environ)

    Returns:
        InputService instance if started, otherwise None
    """
    env = env or {}
    input_cfg = config.get('loggers', {}).get('input', {})
    input_enabled = input_cfg.get('enabled', False) or env.get('ENABLE_INPUT_LOGGER') == '1'

    if not input_enabled:
        logging.info("Input logging not enabled")
        return None

    try:
        service = InputService(config_path=input_cfg.get('config_path'), use_subprocess=input_cfg.get('use_subprocess', True))
        service.start()
        event_mgr.register_source('input', 'logger', {'type': 'InputLogger'})
        logging.info("InputLogger started and registered as 'input'")
        return service
    except Exception as e:
        logging.warning(f"Failed to start InputLogger: {e}")
        return None