class SessionBaselineTracker:
    def __init__(self, min_samples_for_warmup: int = 5):
        self.min_samples_for_warmup = min_samples_for_warmup
        self.samples = []

    def add_sample(self, value):
        self.samples.append(value)

    def is_warmed(self) -> bool:
        return len(self.samples) >= self.min_samples_for_warmup
