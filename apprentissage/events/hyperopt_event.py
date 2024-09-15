from framework.events.generic_event import GenericEvent


class HyperoptEvent(GenericEvent):

    def to_string(self) -> str:
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.entry_mult = None
        # self.exit_mult = None
        # self.future_candles = None
