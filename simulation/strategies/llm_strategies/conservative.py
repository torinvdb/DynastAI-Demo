from .base import LLMStrategy

class ConservativeStrategy(LLMStrategy):
    def __init__(self, **kwargs):
        super().__init__(
            system_prompt=(
                "You are a cautious advisor. Your main goal is to keep all metrics (Piety, Stability, Power, Wealth) within the safe range (1-99) for as long as possible. Avoid risky decisions. Respond with only 'Left' or 'Right'. No explanation needed."
            ),
            **kwargs
        )
    @property
    def name(self):
        return "conservative" 