# Integration smoke test for NetworkLogger using the integration helper
from loggers.network_integration import integrate_network_logger

class FakeEventMgr:
    def __init__(self):
        self.streams = {}
    def register_source(self, source_id, kind, metadata=None):
        self.streams[source_id] = {'kind': kind, 'metadata': metadata}


def test_network_integration_smoke(monkeypatch):
    # Make NetworkService a fake to avoid real subprocesses
    class FakeSvc:
        def __init__(self, config_path=None, use_subprocess=True):
            self.started = False
        def start(self):
            self.started = True
        def stop(self):
            self.started = False
        def is_running(self):
            return self.started

    monkeypatch.setenv('ENABLE_NETWORK_LOGGER', '1')
    monkeypatch.setattr('loggers.network_integration.NetworkService', FakeSvc)

    mgr = FakeEventMgr()
    cfg = {'loggers': {'network': {'enabled': True}}}

    svc = integrate_network_logger(mgr, cfg, env={})
    assert svc is not None and svc.is_running()
    assert 'network' in mgr.streams

    svc.stop()
    monkeypatch.delenv('ENABLE_NETWORK_LOGGER')
