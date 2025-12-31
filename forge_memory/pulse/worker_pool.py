import threading
import time
from queue import Queue, Empty
from typing import Callable, List

class WorkerPool:
    """Spawns N worker threads that call a hypothesis function and push records to their queues."""

    def __init__(self, worker_count: int = 4, queue_size: int = 1000):
        self.worker_count = worker_count
        self.queues: List[Queue] = [Queue(maxsize=queue_size) for _ in range(worker_count)]
        self.threads: List[threading.Thread] = []
        self.running = False

    def start(self, hypothesis_fn: Callable[[int], dict]):
        self.running = True
        for wid in range(self.worker_count):
            t = threading.Thread(target=self._worker_loop, args=(wid, hypothesis_fn), daemon=True)
            t.start()
            self.threads.append(t)

    def _worker_loop(self, wid: int, hypothesis_fn: Callable[[int], dict]):
        q = self.queues[wid]
        while self.running:
            record = hypothesis_fn(wid)
            try:
                q.put(record, timeout=0.1)
            except Exception:
                time.sleep(0.01)

    def stop(self):
        self.running = False
        for t in self.threads:
            t.join(timeout=1.0)

    def get_queues(self) -> List[Queue]:
        return self.queues

    def put_direct(self, wid: int, record: dict):
        self.queues[wid].put(record)
