# Integration smoke test for ActivityLogger using the integration helper
from loggers.activity_integration import integrate_activity_logger

class FakeEventMgr:
    def __init__(self):
        self.streams = {}
    def register_source(self, source_id, kind, metadata=None):
        self.streams[source_id] = {'kind': kind, 'metadata': metadata}


def test_activity_integration_smoke(monkeypatch):
    # Make ActivityService a fake to avoid real subprocesses
    class FakeSvc:
        def __init__(self, config_path=None, use_subprocess=True):
            self.started = False
        def start(self):
            self.started = True
        def stop(self):
            self.started = False
        def is_running(self):
            return self.started

    monkeypatch.setenv('ENABLE_ACTIVITY_LOGGER', '1')
    monkeypatch.setattr('loggers.activity_integration.ActivityService', FakeSvc)

    mgr = FakeEventMgr()
    cfg = {'loggers': {'activity': {'enabled': True}}}

    svc = integrate_activity_logger(mgr, cfg, env={})
    assert svc is not None and svc.is_running()
    assert 'activity' in mgr.streams

    svc.stop()
    monkeypatch.delenv('ENABLE_ACTIVITY_LOGGER')
