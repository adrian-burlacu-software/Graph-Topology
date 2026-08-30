
from __future__ import annotations


class Realizer:
    """
    Final language surface.

    The architecture chooses the content before this stage.
    """

    FALLBACKS={
        "social_greeting":"Hello!",
        "social_thanks":"You're welcome!",
        "social_goodbye":"Bye!",
        "social_affection":"That's nice to hear.",
        "explore_assistant":"I'm thinking about our conversation.",
        "request_information":"I'm not sure yet.",
        "request_explanation":"I'm not sure yet.",
        "request_opinion":"I'm not sure yet.",
        "challenge_claim":"I'm not certain.",
        "request_generation":"Sure.",
        "request_action":"Sure. What would you like me to do?",
        "continue_conversation":"Tell me more.",
    }

    def fallback(self,goal):
        return self.FALLBACKS.get(
            goal.name,
            "I'm not sure yet.",
        )
