"""
The implementation of the Thought Rollback (TR).
"""

import reasoner
import thought_model
import thought_prompter
import visualization
import tr_thought_prompts
import tr_system_prompts


from iclp.old.pipeline import Pipeline
from iclp.old.model import define_model

# from llmpebase.prompt import get_system_prompts, get_thought_prompts


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
    # data_config = cfg.get("data", {})

    # system_prompts = get_system_prompts(data_config)
    system_prompts = tr_system_prompts.RollbackSystemPrompts()
    # thought_prompts = get_thought_prompts(data_config)
    thought_prompts = tr_thought_prompts.BaseRollbackThoughtPrompts()
    llm_model = define_model(model_config=model_config)
    prompter = thought_prompter.TRStructurePrompt(
        system_prompts=system_prompts,
        thought_prompts=thought_prompts,
        rollback_prompts=tr_thought_prompts.RollbackPrompts(),
    )
    llm_thought = thought_model.TRThoughtModel(
        llm_model=llm_model, model_config=model_config, prompter=prompter
    )

    tr_reasoner = reasoner.ThoughtRollbackReasoner(
        thought_model=llm_thought,
        model_config=model_config,
        logging_config=logging_config,
        visualizer=visualization.TRVisualizer(logging_config=logging_config),
    )

    pipeline = Pipeline(
        reasoner=tr_reasoner,
    )
    pipeline.setup()
    pipeline.load_data()
    pipeline.execute()


if __name__ == "__main__":
    _main()
