# Unit tests for Network integration
from loggers.network_integration import integrate_network_logger

class FakeEventMgr:
    def __init__(self):
        self.streams = {}
    def register_source(self, source_id, kind, metadata=None):
        self.streams[source_id] = {'kind': kind, 'metadata': metadata}


def test_integrate_network_logger_enabled(monkeypatch):
    fake_mgr = FakeEventMgr()
    cfg = {'loggers': {'network': {'enabled': True, 'use_subprocess': False}}}

    class FakeSvc:
        def __init__(self, config_path=None, use_subprocess=True):
            self.started = False
        def start(self):
            self.started = True
        def stop(self):
            self.started = False
    monkeypatch.setattr('loggers.network_integration.NetworkService', FakeSvc)

    svc = integrate_network_logger(fake_mgr, cfg, env={})
    assert svc is not None
    assert svc.started is True
    assert 'network' in fake_mgr.streams


def test_integrate_network_logger_disabled():
    fake_mgr = FakeEventMgr()
    cfg = {'loggers': {'network': {'enabled': False}}}
    svc = integrate_network_logger(fake_mgr, cfg, env={})
    assert svc is None
    assert 'network' not in fake_mgr.streams
