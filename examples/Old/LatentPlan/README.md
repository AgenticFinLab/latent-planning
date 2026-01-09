# Instruction

This is an instruction about how to run the code. The Python version must be higher than 3.11.

## Naming 

See the README.md of the projinit package.


### Plan Synthetic

At this stage, we primarily use Direct Synthetic (DSynt), which requires decomposing the answers of all samples and then summarizing plans from the individual steps.

Using the following command lines to perform the data synthetic:

- Math:
  - ```python examples/LatentPlan/data_synthesize.py -c  examples/LatentPlan/configs/MATH/deepseek_direct-synthesize_plan.yml -b LatentPlanResult```

  - ```python examples/LatentPlan/data_synthesize.py -c  examples/LatentPlan/configs/MATH/gpt4o_direct-synthesize_plan.yml -b LatentPlanResult```

- Code Generation:
  - ```python examples/LatentPlan/data_synthesize.py -c  examples/LatentPlan/configs/HumanEval/deepseek_direct-synthesize_plan.yml -b LatentPlanResult```

  - ```python examples/LatentPlan/data_synthesize.py -c  examples/LatentPlan/configs/HumanEval/gpt4o_direct-synthesize_plan.yml -b LatentPlanResult```


Check the synthesized data located in the folder specified by `synthesized_path/synthesized_name` in the configuration file.

Upload the synthesized data presented in `hf_train_filename` and `hf_test_filename` to the huggingface website with the name `hf_dataname` presented in the configuration file. 


### Data Preparation

We need to download the dataset from the huggingface website and then prepare the dataset for the finetuning process. Run the command

```
python examples/LatentPlan/data_prepare.py -c examples/LatentPlan/configs/MATH/gpt4o-DSynt_data_to_hf.yml -b ExptsPlanFT
```
 
This yields 8 files containing:
  1. __explicit plans__, Samples whose `assistant part` is the next explicit plan. 
    - The obtained train and test files are defined by `ft_train_e_dataname`, `ft_test_e_dataname` under the provided config.
  2. __implicit plans__, Samples whose `assistant part` is the next implicit plan. 
    - The obtained train and test files are defined by `ft_train_dataname`, `ft_test_dataname` under the provided config.
  3. __reason with explicit plans__, Samples whose `assistant part` is the whole reasoning process guided by explicit plans: plan -> step -> plan -> step.... 
    - The obtained train and test files are defined by `ft_train_e_plan_reason_dataname`, `ft_test_e_plan_reason_dataname` under the provided config.
  4. __reason with implicit plans__, Samples whose `assistant part` is the whole reasoning process guided by implicit plan: -> step -> plan -> step.... 
    - The obtained train and test files are defined by `ft_train_plan_reason_dataname`, `ft_test_plan_reason_dataname` under the provided config.

Note that these files are stored under the `synthesized_path/synthesized_name` where these two names are defined in the provided config file.

The 1, 2 is for the concept learner that build the latent concept plan space. The 3,4 is for the generator.

Each sample in the prepared data is formatted to contain a `message` that is a dict presenting the user content and the assistant content, which is the most common structure:
```json
[
  {
    "role": "user",
    "content": "...."
  },
  {
    "role": "assistant",
    "content": "...."
  }
]
```


### Model Fine-tuning

#### Input Template

First, we must know that the input of the decoder-only models should contain the whole text including system prompt (optional), user's prompt --- question or any instructions, assistant content --- groundtruth responses from the LLMs. Thus, during the training, the model is to predict the next-work of each token of the input. This is why the loss is computed based on the input text and 1-right-shift of the input text (if you finetune on the whole text but in some cases, you only want to finetune on the assistant content, see below).

Each model provided in the HuggingFace website has its own requirement of input template presenting the keywords and structures used during the provider trained the model. Thus, we MUST convert our own sample to be the format same as the model's format. For example:

<table>
  <tr>
    <th></th>
    <th>instruction_part</th>
    <th>response_part</th>
  </tr>
  <tr>
    <td><strong>Llama-3.2</strong></td>
    <td><|start_header_id|>user<|end_header_id|>\n\n</td>
    <td><|start_header_id|>assistant<|end_header_id|>\n\n</td>
  </tr>
  <tr>
    <td><strong>Qwen2.5</strong></td>
    <td><|im_start|>user\n</td>
    <td><|im_start|>assistant\n</td>
  </tr>
</table>

Therefore, we need to perform `tokenizer.apply_chat_template` to convert our prepared input to the format aligned with the model. Within this function, the `add_generation_prompt` should be set to the False during the training as desired responses of the model has been included in the input and you do not want the model to generate a response at this stage. During the inference stage, we should set `add_generation_prompt=True` because you want it to predict the next token in the sequence.



#### Operations -- Finetuning on Plan Generation

With the `TrainPlan.json` and `TestPlan.json`, we are able to fine-tune the models. At this stage, we support the fine-tuning of the models, including: 

- Llama models: `Llama3-2-1B`, `Llama3-2-3B`.
- Qwen models: `Qwen2.5-0.5B`, `Qwen2.5-1.5B`, `Qwen2.5-3B`, `Qwen2.5-7B`, `Qwen2.5-14B`.

1. Run the command:

```
python examples/LatentPlan/finetune_main.py -c *.yml -b *
```

where two * present the configuration file and the project name, respectively.
Here are the examples:

- Llama: 
    - latent plan
        - ```python examples/LatentPlan/main_finetune_plan.py -c examples/LatentPlan/configs/MATH/Llama3/Llama3-2-1B_FT_GPT4o-DSynt-Latent.yml -b ICMLPlan```
        - ```python examples/LatentPlan/main_finetune_plan.py -c examples/LatentPlan/configs/MATH/Llama3/Llama3-2-3B_FT_GPT4o-DSynt-Latent.yml -b ICMLPlan```
    - explicit plan
        - ```python examples/LatentPlan/main_finetune_plan.py -c examples/LatentPlan/configs/MATH/Llama3/Llama3-2-1B_FT_on_DSynt-explicit.yml -b ICMLPlan```
        - ```python examples/LatentPlan/main_finetune_plan.py -c examples/LatentPlan/configs/MATH/Llama3/Llama3-2-3B_FT_on_DSynt-explicit.yml -b ICMLPlan```

- Qwen2.5:
    - latent plan
        - ```python examples/LatentPlan/main_finetune_plan.py -c examples/LatentPlan/configs/MATH/Qwen/Qwen2.5-0.5B_FT_on_GPT4o-DSynt.yml -b ICMLPlan```
        - ```python examples/LatentPlan/main_finetune_plan.py -c examples/LatentPlan/configs/MATH/Qwen/Qwen2.5-1.5B_FT_on_GPT4o-DSynt.yml -b ICMLPlan```
        - ```python examples/LatentPlan/main_finetune_plan.py -c examples/LatentPlan/configs/MATH/Qwen/Qwen2.5-3B_FT_on_GPT4o-DSynt.yml -b ICMLPlan```
        - ```python examples/LatentPlan/main_finetune_plan.py -c examples/LatentPlan/configs/MATH/Qwen/Qwen2.5-7B_FT_on_GPT4o-DSynt.yml -b ICMLPlan```
        - ```python examples/LatentPlan/main_finetune_plan.py -c examples/LatentPlan/configs/MATH/Qwen/Qwen2.5-14B_FT_on_GPT4o-DSynt.yml -b ICMLPlan```
    - explicit plan
        - ```python examples/LatentPlan/main_finetune_plan.py -c examples/LatentPlan/configs/MATH/Qwen/Qwen2.5-0.5B_FT_on_DSynt-explicit.yml -b ICMLPlan```
        - ```python examples/LatentPlan/main_finetune_plan.py -c examples/LatentPlan/configs/MATH/Qwen/Qwen2.5-1.5B_FT_on_DSynt-explicit.yml -b ICMLPlan```
        - ```python examples/LatentPlan/main_finetune_plan.py -c examples/LatentPlan/configs/MATH/Qwen/Qwen2.5-3B_FT_on_DSynt-explicit.yml -b ICMLPlan```
        - ```python examples/LatentPlan/main_finetune_plan.py -c examples/LatentPlan/configs/MATH/Qwen/Qwen2.5-7B_FT_on_DSynt-explicit.yml -b ICMLPlan```
        - ```python examples/LatentPlan/main_finetune_plan.py -c examples/LatentPlan/configs/MATH/Qwen/Qwen2.5-14B_FT_on_DSynt-explicit.yml -b ICMLPlan```

This yields the model saved to the `log_config["checkpoint_path"]` that is `ICMLPlan/checkpoint/*` where * present the corresponding folder. Besides, all results and training details are shared on the wandb website.



#### Operations -- Finetuning for Plan Concept Learner

This is to finetune the concept learner model toward learning a concept space of the plan.

- Llama: 
    - latent plan
        - ```python examples/LatentPlan/main_finetune_concept.py -c examples/LatentPlan/configs/MATH/Llama3Concept/Llama3-2-1B_FT_on_GPT4o-DSynt-Latent.yml -b ICMLPlan```

- Qwen2.5:
    - latent plan
        - ```python examples/LatentPlan/main_finetune_concept.py -c examples/LatentPlan/configs/MATH/QwenConcept/Qwen2.5-0.5B_all-MiniLM-L6-v1_FT_on_GPT4o-DSynt-Latent.yml -b ICMLPlan```


#### Key Parameters -- Before continuing

Before moving to the finetuning of the models toward plan-based reasoning, we should understand some parameter involved in the config file.

1. Under the `train` of the config file, there are two important parameters: `plan_type` and `reason_plan_space`. In detail:
    - `plan_type`: `explicit` (the plan shows the problem-related guidance) or `implicit` (the plan shows the generalized guidance).
    - `reason_plan_space`: `language` (reasoning based on the language plan) or `latent` (reasoning based on the latent plan learned by the concept learner)

Setting these two parameters will control how to fine-tune the model. Once `reason_plan_space` is set to be `latent`, there must be concept learner fine-tuned on either `explicit` or `implicit` provided under the `learner` as shown below.

2. Under the `model` of the config file, there are these parameters needed to be set:
    - `learner`: Present the config file used to finetune the concept learner included in `learner_ckpt` below.
    - `reasoner`:
      - `learner_ckpt`: the path of the checkpoint of the learner to be used
      - `model_name` and `model_type` to define the decoder-only model
      - `chat_template` and `system_message` 



#### Operations -- Finetuning for Plan-based reasoning in the language space

This is to finetune the model toward performing the plan-based reasoning in the language space. Set `reason_plan_space: language`:

- Llama: 
    - explicit+language plan
        - ```python examples/LatentPlan/main_finetune_reasoner.py -c examples/LatentPlan/configs/MATH/Llama3Concept/Llama3-2-1B_FT_on_GPT4o-DSynt-Latent.yml -b ICMLPlan```

- Qwen2.5:
    - explicit+language plan
        - ```python examples/LatentPlan/main_finetune_reasoner.py -c examples/LatentPlan/configs/MATH/QwenReasoner/Qwen2.5-0.5B--all-MiniLM-L6-v1--GPT4o-DSynt--ELanguage.yml -b ExptsPlanFT -p Reasoner -u "train|plan_type;train|reason_plan_space" -r experiments.csv```
    - implicit+language plan
        - ```python examples/LatentPlan/main_finetune_reasoner.py -c examples/LatentPlan/configs/MATH/QwenReasoner/Qwen2.5-0.5B--all-MiniLM-L6-v1--GPT4o-DSynt--ILanguage.yml -b ExptsPlanFT -p Reasoner -u "train|plan_type;train|reason_plan_space" -r experiments.csv```

where the `ILatent` derives from the implicit (I) latent.


#### Operations -- Finetuning for Plan-based reasoning in the latent space

This is to finetune the generator model toward performing the plan-based reasoning.


Here are the command lines to run:

- Llama: 
    - latent plan
        - ```python examples/LatentPlan/main_finetune_generator.py -c examples/LatentPlan/configs/MATH/Llama3Reasoner/Llama3-2-1B_Llama3-2-1B_FT_on_GPT4o-DSynt-Latent.yml -b ICMLPlan```

- Qwen2.5:
    - latent plan
        - ```python examples/LatentPlan/main_finetune_generator.py -c examples/LatentPlan/configs/MATH/QwenReasoner/Qwen2.5-0.5B--all-MiniLM-L6-v1--GPT4o-DSynt--ILatent.yml -b ICMLPlan```




### Model Evaluation 

Here are two types of model evaluation. 

1. The first one is to compute the loss of the fine-tuned model on the test set, which has the same structure of the train set.

For example, to test the finetuned Llama3 model for the MATH dataset, 
```
python examples/LatentPlan/inference_main.py -c examples/LatentPlan/configs/MATH/Llama3/Llama3-2-1B_FT_on_DSynt.yml -b ICMLPlan
```
In the configuration file, you need to set:
1. `model_name` under the `model` block so that the sample template can be determined
2. `finetuned_model_path` and `model_folder_name` under the `evaluation` block to set the model to be evaluated
3. `result_path` under `logging` to determine where to save the eval results


2. The second is to evaluate the model which is trained to generate the plan for reasoning, on the real reasoning process.



## Common Q&A
1. Why the vocabulary size get from the `self.decoder.config.vocab_size` is different from that 
from the `len(tokenizer)`. Generally vocab_size of the model is larger than the that of the tokenizer.
   - This inconsistency is due to a mechanism that the embedding layer of the model will be given a size larger than the vocabulary. These additional space/embeddings are never used and is just for hardware efficiency, i.e., with an input, the sum of their scores may be e-10. Another mechanism is that: 1). Once a new token is added to the tokenizer, the size of the embedding layer will increase 1, 2). the embedding of this token is accessed based on its id in the tokenizer, which is not the new added embedding, meaning that the model will activate the old pre-defined embedding for this new token. See. https://github.com/huggingface/transformers/issues/4875.

