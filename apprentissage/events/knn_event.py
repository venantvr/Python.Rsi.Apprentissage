from framework.events.generic_event import GenericEvent


class KnnEvent(GenericEvent):

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def to_string(self) -> str:
        return f'[{self['confidence']}% {self['trend']}]' if self is not None else None
