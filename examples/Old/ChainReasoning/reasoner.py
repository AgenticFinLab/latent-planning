"""
A reasoner to perform the reasoning step by step in a chain structure.
"""

from iclp.old.model.thought_structure import chains
from iclp.old.model.thought_structure.base import BaseThoughtStructure
from iclp.old.reasoner.structured_thought import StructuredThoughtReasoner


class ChainThoughtReasoner(StructuredThoughtReasoner):
    """
    A CoT reasoner to answer the question with the request model.
    """

    def define_structure(self) -> type[BaseThoughtStructure]:
        return chains.ChainThoughtStructure(
            thought_model=self.thought_model,
            model_config=self.model_config,
            logging_config=self.logging_config,
            visualizer=self.visualizer,
        )
