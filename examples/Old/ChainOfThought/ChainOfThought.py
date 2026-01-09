"""
Implementation of Chain Of Thought [1].

[1]. Wei, et.al., Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, 23.
"""

from iclp.old.pipeline import Pipeline
from iclp.old.reasoner import direct_llm
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

    model_config = cfg.get("model", {})
    cot_reasoner = direct_llm.BaseLLMReasoner(model_config=model_config)

    pipeline = Pipeline(reasoner=cot_reasoner)
    pipeline.setup()
    pipeline.load_data()
    pipeline.execute()


if __name__ == "__main__":
    _main()
