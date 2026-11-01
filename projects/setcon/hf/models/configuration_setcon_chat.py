import copy

from .configuration_internvl_chat import InternVLChatConfig


class SetConChatConfig(InternVLChatConfig):
    model_type = 'setcon_chat'

    def __init__(self, template=None, grounding_encoder_type='sam3', **kwargs):
        super().__init__(**kwargs)
        if template is not None:
            self.template = template
        self.grounding_encoder_type = grounding_encoder_type

    def to_dict(self):
        output = copy.deepcopy(super().to_dict())
        output['model_type'] = self.__class__.model_type
        output['template'] = self.template
        output['grounding_encoder_type'] = self.grounding_encoder_type
        return output
