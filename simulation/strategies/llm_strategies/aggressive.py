from .base import LLMStrategy

class AggressiveStrategy(LLMStrategy):
    def __init__(self, **kwargs):
        super().__init__(
            system_prompt=(
                "You are a bold ruler. Take risks to maximize gains, even if it means some metrics may get close to the edge. Respond with only 'Left' or 'Right'. No explanation needed."
            ),
            **kwargs
        )
    @property
    def name(self):
        return "aggressive" 