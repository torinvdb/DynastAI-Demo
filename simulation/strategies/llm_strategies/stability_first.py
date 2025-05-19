from .base import LLMStrategy

class PeopleFirstStrategy(LLMStrategy):
    def __init__(self, **kwargs):
        super().__init__(
            system_prompt=(
                "You are a populist ruler. Always prioritize the Stability metric above all others, but still try to survive as long as possible. Respond with only 'Left' or 'Right'. No explanation needed."
            ),
            **kwargs
        )
    @property
    def name(self):
        return "stability_first" 