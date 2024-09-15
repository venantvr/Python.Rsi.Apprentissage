from framework.events.generic_event import GenericEvent


class RegressionEvent(GenericEvent):

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def to_string(self) -> str:
        return f'[{round(self['percentage'], 2)}% sur {self['value']}]' if self is not None else None
