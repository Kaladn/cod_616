from loggers.process_integration import integrate_process_logger


class FakeEventMgr:
    def __init__(self):
        self.streams = {}
    def register_source(self, source_id, kind, metadata=None):
        self.streams[source_id] = {'kind': kind, 'metadata': metadata}


def test_process_integration_smoke(monkeypatch):
    class FakeSvc:
        def __init__(self, config_path=None, use_subprocess=True):
            self.started = False
        def start(self):
            self.started = True
        def stop(self):
            self.started = False
        def is_running(self):
            return self.started

    monkeypatch.setenv('ENABLE_PROCESS_LOGGER', '1')
    monkeypatch.setattr('loggers.process_integration.ProcessService', FakeSvc)

    mgr = FakeEventMgr()
    cfg = {'loggers': {'process': {'enabled': True}}}

    svc = integrate_process_logger(mgr, cfg, env={})
    assert svc is not None and svc.is_running()
    assert 'process' in mgr.streams

    svc.stop()
    monkeypatch.delenv('ENABLE_PROCESS_LOGGER')