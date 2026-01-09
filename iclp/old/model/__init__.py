"""
An interface of models and prompts
"""

import logging

from trlm.model.prompting import (
    base,
    fewshot,
    cot,
    zeroshot_cot,
    zeroshot,
)


prompts_factory = {
    "mmlu": {
        "fewshot": fewshot.MMLUFewShotPrompting,
        "cot": cot.MMLUCoTPrompting,
        "zeroshot_cot": zeroshot_cot.MMLUZeroShotCoTPrompting,
        "zeroshot": zeroshot.MMLUZeroShotPrompting,
    },
    "gsm8k": {
        "fewshot": base.BasePrompting,
        "cot": cot.GSM8KCoTPrompting,
        "zeroshot_cot": base.BaseZeroShotCoTPrompting,
        "zeroshot": base.BaseZeroShotPrompting,
    },
    "gameof24": {
        "fewshot": "Not provided",
        "cot": "Not provided",
        "zeroshot_cot": zeroshot_cot.GameOf24ZeroShotCoTPrompting,
        "zeroshot": zeroshot.GameOf24ZeroShotPrompting,
    },
    "math": {
        "fewshot": fewshot.ProblemFewShotPrompting,
        "cot": cot.MATHCoTPrompting,
        "zeroshot_cot": base.BaseZeroShotCoTPrompting,
        "zeroshot": base.BaseZeroShotPrompting,
    },
    "bbh": {
        "fewshot": fewshot.ProblemFewShotPrompting,
        "cot": cot.BBHCoTPrompting,
        "zeroshot_cot": base.BaseZeroShotCoTPrompting,
        "zeroshot": base.BaseZeroShotPrompting,
    },
    "theoremqa": {
        "fewshot": fewshot.TheoremQAFewShotPrompting,
        "cot": cot.TheoremQACoTPrompting,
        "zeroshot_cot": zeroshot_cot.TheoremQAZeroShotCoTPrompting,
        "zeroshot": zeroshot.TheoremQAZeroShotPrompting,
    },
    "csqa": {
        "zeroshot_cot": zeroshot_cot.CSQAZeroShotCoTPrompting,
        "zeroshot": zeroshot.CSQAZeroShotPrompting,
    },
    "aqua": {
        "fewshot": fewshot.AQUAFewShotPrompting,
        "zeroshot_cot": zeroshot_cot.AQUAZeroShotCoTPrompting,
        # CoT of AQUA is the same as the fewshot as the
        # 'rationale' is provided by the dataset
        "cot": fewshot.AQUAFewShotPrompting,
        "zeroshot": zeroshot.AQUAZeroShotPrompting,
    },
    "svamp": {
        "fewshot": fewshot.ProblemFewShotPrompting,
        "zeroshot_cot": base.BaseZeroShotCoTPrompting,
        "zeroshot": base.BaseZeroShotPrompting,
    },
}


def define_prompt(data_config: dict, model_config: dict):
    """Define the datasets based on the config file."""
    data_name = data_config["data_name"].lower()
    prompt_type = model_config["prompt_type"].lower()
    logging.info("Defined %s Prompting for %s .", prompt_type, data_name)

    return prompts_factory[data_name][prompt_type](model_config)
