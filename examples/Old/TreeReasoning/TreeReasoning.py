"""
A reasoning process organized as a tree thought structure.
"""

import reasoner

from iclp.old.pipeline import Pipeline
from iclp.old.model import define_model
from iclp.old.model.thought_structure import thought_model


from iclp.old.util.tools import ConfigLoader
import yaml
import argparse


def _main():
    """The core function for model running."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        dest="config_path",
        default="",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=ConfigLoader) or {}

    # Set the basic llm model to be used by each component
    model_config = cfg.get("model", {})
    logging_config = cfg.get("logging", {})

    llm_model = define_model(model_config=model_config)

    llm_thought = thought_model.LlmThoughtModel(llm_model=llm_model)

    chain_reasoner = reasoner.TreeThoughtReasoner(
        thought_model=llm_thought,
        model_config=model_config,
        logging_config=logging_config,
    )

    pipeline = Pipeline(
        reasoner=chain_reasoner,
    )
    pipeline.setup()
    pipeline.load_data()
    pipeline.execute()


if __name__ == "__main__":
    _main()
