import types
from loggers.activity_integration import integrate_activity_logger


class FakeEventMgr:
    def __init__(self):
        self.streams = {}
    def register_source(self, source_id, kind, metadata=None):
        self.streams[source_id] = {'kind': kind, 'metadata': metadata}


def test_integrate_activity_logger_enabled(monkeypatch):
    fake_mgr = FakeEventMgr()
    cfg = {'loggers': {'activity': {'enabled': True, 'use_subprocess': False}}}

    # Patch ActivityService to a fake that records start/stop
    class FakeSvc:
        def __init__(self, config_path=None, use_subprocess=True):
            self.started = False
        def start(self):
            self.started = True
        def stop(self):
            self.started = False
    monkeypatch.setattr('loggers.activity_integration.ActivityService', FakeSvc)

    svc = integrate_activity_logger(fake_mgr, cfg, env={})
    assert svc is not None
    assert svc.started is True
    assert 'activity' in fake_mgr.streams


def test_integrate_activity_logger_disabled():
    fake_mgr = FakeEventMgr()
    cfg = {'loggers': {'activity': {'enabled': False}}}
    svc = integrate_activity_logger(fake_mgr, cfg, env={})
    assert svc is None
    assert 'activity' not in fake_mgr.streams
