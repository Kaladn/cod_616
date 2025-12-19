from loggers.input_integration import integrate_input_logger


class FakeEventMgr:
    def __init__(self):
        self.streams = {}
    def register_source(self, source_id, kind, metadata=None):
        self.streams[source_id] = {'kind': kind, 'metadata': metadata}


def test_input_integration_smoke(monkeypatch):
    # Make InputService a fake to avoid real subprocesses
    class FakeSvc:
        def __init__(self, config_path=None, use_subprocess=True):
            self.started = False
        def start(self):
            self.started = True
        def stop(self):
            self.started = False
        def is_running(self):
            return self.started

    monkeypatch.setenv('ENABLE_INPUT_LOGGER', '1')
    monkeypatch.setattr('loggers.input_integration.InputService', FakeSvc)

    mgr = FakeEventMgr()
    cfg = {'loggers': {'input': {'enabled': True}}}

    svc = integrate_input_logger(mgr, cfg, env={})
    assert svc is not None and svc.is_running()
    assert 'input' in mgr.streams

    svc.stop()
    monkeypatch.delenv('ENABLE_INPUT_LOGGER')
