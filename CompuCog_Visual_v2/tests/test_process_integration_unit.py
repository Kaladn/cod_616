from loggers.process_integration import integrate_process_logger


class FakeEventMgr:
    def __init__(self):
        self.streams = {}
    def register_source(self, source_id, kind, metadata=None):
        self.streams[source_id] = {'kind': kind, 'metadata': metadata}


def test_integrate_process_logger_enabled(monkeypatch):
    fake_mgr = FakeEventMgr()
    cfg = {'loggers': {'process': {'enabled': True, 'use_subprocess': False}}}

    class FakeSvc:
        def __init__(self, config_path=None, use_subprocess=True):
            self.started = False
        def start(self):
            self.started = True
        def stop(self):
            self.started = False
    monkeypatch.setattr('loggers.process_integration.ProcessService', FakeSvc)

    svc = integrate_process_logger(fake_mgr, cfg, env={})
    assert svc is not None
    assert svc.started is True
    assert 'process' in fake_mgr.streams


def test_integrate_process_logger_disabled():
    fake_mgr = FakeEventMgr()
    cfg = {'loggers': {'process': {'enabled': False}}}
    svc = integrate_process_logger(fake_mgr, cfg, env={})
    assert svc is None
    assert 'process' not in fake_mgr.streams
