from .base import LLMStrategy

class ReligiousStrategy(LLMStrategy):
    def __init__(self, **kwargs):
        super().__init__(
            system_prompt=(
                "You are a devout leader. Always prioritize the Piety metric above all others, but still try to survive as long as possible. Respond with only 'Left' or 'Right'. No explanation needed."
            ),
            **kwargs
        )
    @property
    def name(self):
        return "religious" 