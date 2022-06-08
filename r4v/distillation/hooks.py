from ..utils import get_subclasses


class DistillationHook:
    def __init__(self, config):
        self.config = config

    def initialize(self, **kwargs):
        pass

    def pre_epoch(self, **kwargs):
        pass

    def post_epoch(self, **kwargs):
        pass

    def pre_training(self, **kwargs):
        pass

    def post_training(self, **kwargs):
        pass

    def pre_validation(self, **kwargs):
        pass

    def post_validation(self, **kwargs):
        pass


def build_hook(config):
    subclasses = {c.__name__: c for c in get_subclasses(DistillationHook)}
    hook = subclasses[config["type"]](config)
    return hook
