import copy

from .configuration_internvl_chat import InternVLChatConfig


class SetConChatConfig(InternVLChatConfig):
    model_type = 'setcon_chat'

    def __init__(self, template=None, **kwargs):
        super().__init__(**kwargs)
        if template is not None:
            self.template = template

    def to_dict(self):
        output = copy.deepcopy(super().to_dict())
        output['model_type'] = self.__class__.model_type
        output['template'] = self.template
        return output
