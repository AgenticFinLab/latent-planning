"""
Prompts related to the plan-based reasoning.
"""

from trlm.prompt.thought_prompt import BaseThoughtPrompts, ThoughtGenerationPrompts
from trlm.prompt.generic import BasicThoughtPromptFormat
from trlm.prompt.system_prompt import BaseSystemPrompts


class PlanSystemPrompts(BaseSystemPrompts):
    """A series of system prompts for the plan operation."""

    answer_decompose_prompt = "You are an expert in breaking down an answer to a telecommunication question into multiple logical reasoning steps without changing the answer's words or sentences. Carefully read the given question and the answer, then decompose the answer into individual reasoning steps. Ensure each step is not small and thus contains a complete computation, analysis, or design process. Each step should be strictly extracted from the given answer without making any changes. Please only add the Step ID before each decomposed step, such as 'Step 1:' or 'Step 2:'."

    plan_summarization_prompt: str = (
        """You are an expert in identifying, extracting, and summarizing the plan that underpins one reasoning step. The summarized plan should be a general-purpose reasoning instruction and, thus, is a high-level, question-agnostic principle. Please get such a plan containing the highest-level ideas, principles, rules, or theorems from the given reasoning step. Start by reviewing the given question and any previous reasoning steps already taken along with their corresponding plans, then directly summarize the plan of the given reasoning step. Please summarize the plan directly and briefly, avoiding including the specific contents of the given question or any reasoning steps."""
    )

    plan_exclusion_generation_prompt: str = (
        """As an expert in problem-solving, you are adept at methodical, step-by-step reasoning while avoiding duplicating the given plans, each presenting a general-purpose reasoning instruction for one step. You need to know that each plan is a high-level, question-agnostic principle that facilitates deducing a single logical reasoning step toward addressing one task. Thus, excluding the plan means having a new and different plan to generate the corresponding next step. Remember, your response should only include one next step. Start by reviewing the problem and reasoning steps, then exclude the given specific plans to generate the next step. You can ignore the plan exclusion when no plan is given. The next step should contain the precise analysis and the corresponding mathematical expression. Utilize Python Programming as an auxiliary tool when necessary."""
    )

    thought_plan_assessment_prompt: str = (
        """You are a professional mathematician with expertise in assessing a plan that presents general-purpose reasoning instruction for generating the next reasoning step. Specifically, the plan is a high-level, question-agnostic principle that facilitates deducing a single logical reasoning step toward addressing one task. You should assess the plan by scoring it based on whether it guides generating the reasonable reasoning step that progresses the problem-solving. Start by reviewing the given problem, reasoning steps already taken, and the generated next step guided by the plan, then directly assess this given plan. Importantly, the generated reasoning step guided by this plan is also given to facilitate the assessment. Utilize Python Programming as an auxiliary tool when necessary. The output should be a float score without including any other content. """
    )

    plan_comparison_prompt: str = (
        """As a professional plan comparison expert, your expertise lies in judging whether a plan exists in a plan pool containing various plans. Remember that plan is a general-purpose reasoning instruction and is a high-level, question-agnostic principle. Please perform the comparison in terms of the logic, high-level ideas, theorems, or rules. Please compare the given plan with each of the plans in the pool. Once there is a similar one, return True. Start by reviewing plans in the pool, then directly judge whether the given plan already exists. The output should be either True or False."""
    )

    # This generation prompt contain the plan of each step.
    # generation_prompt: str = (
    #     """You are an expert in solving mathematical problems using methodical, step-by-step reasoning, with each response containing one step. Please only generate one next step in the response. Start by reviewing the given problem and reasoning steps along with their corresponding plans already taken, and then proceed to provide the next step directly. As each step's plan presents its general-purpose reasoning instruction, reviewing these steps and their plans can be a base for generating the next step. Please generate the next step directly, containing the necessary analysis and the corresponding specific mathematical expression. Utilize Python Programming as an auxiliary only when necessary and report the output in the reasoning step. Note do not only perform analysis, the generated next reasoning step must contain the specific progress."""
    # )

    plan_guided_generation_prompt: str = (
        """As an expert in problem-solving, you are skilled in methodical, step-by-step reasoning guided by plans, each presenting a general-purpose reasoning instruction for one step. The plan is a high-level, question-agnostic instruction guiding the generation of a specific reasoning step toward addressing the question. After reviewing the problem and the reasoning steps taken so far, please follow the given plan to generate the next reasoning step directly."""
    )

    plan_generation_prompt: str = (
        """You are an expert reasoning planner who excels at generating high-level plans to guide the generation of the next steps when using step-by-step reasoning to solve problems. A plan behaves as the conceptual, abstract, and high-level instruction of generating the next reasoning step. After reviewing the given question and existing previous reasoning steps, please generate the plan that can guide the generation of the next reasoning step."""
    )


class PlanPrompts:
    """A series of prompts for the plan generation and creation."""

    # HOLD FOR POSSIBLE FUTURE USE
    # plan_summarization_start_flag: str = """<Plan Summarization>"""
    # plan_summarization_end_flag: str = """<\\Plan Summarization>"""

    plan_exclusion_start_flag: str = """<Exclusion plans>"""
    plan_exclusion_end_flag: str = """<\\Exclusion plans>"""

    plan_assessment_start_flag: str = """<Plan Assessment>"""
    plan_assessment_end_flag: str = """<\\Plan Assessment>"""

    plan_comparison_start_flag: str = """<Plan Pool>"""
    plan_comparison_end_flag: str = """<\\Plan Pool>"""

    plan_chain_start_flag: str = """<Plan Chain>"""
    plan_chain_end_flag: str = """<\\Plan Chain>"""

    plan_start_flag: str = """<Plan>"""
    plan_end_flag: str = """<\\Plan>"""

    # The head of each step
    plan_head: str = "Plan {}."

    step_start_flag: str = """<Step>"""
    step_end_flag: str = """<\\Step>"""

    answer_decompose_prompt = """
    Please decompose the given answer into logical reasoning steps. Please ensure:
    1). Each step should not be small but large enough to only present the complete logic and contain a complete computation, analysis, or reasoning. 
    2). Use as few reasoning steps as possible but it is unacceptable if a step contains too much content.
    3). Be careful not to make each step so small that it contains only a single calculation or a simple statement.
    4). The content of each step should directly be extracted from the given answer without making any changes.
    5). Do not change the words or sentences of the answer while decomposing it into steps.
    6). Ensure the decompose steps contain all content of the original given answer.
    """
    step_plan_summary_prompt = """Let's summarize the plan of the Step {} and directly generate the plan, which is a brief, high-level, question-agnostic principle, without including any question or reasoning step content."""

    # Braces: 1). Question, 2) First Reasoning Step
    first_plan_summarization_prompt = BasicThoughtPromptFormat(
        head="{}\nFor the given question, let's focus on summarize the plan that underpins the first reasoning step.\n",
        content="\n{}\n\n",
        target="Please review Step 1 within {} and summarize its plan, i.e., Plan 1.",
        notice=" Only direct output summarized plan. Do not include the Plan index in the output. Remember that the plan is a high-level, question-agnostic principle. Do not include any question or reasoning step content in the plan.",
        tail="",
        prompt="",
    )

    # Braces: 1). Question, 2) Step index,
    # 3) Reasoning Chain, 4) Plan Chain, 5) Plan thought
    # 6) Chain flag, 7) Plan flag Step index, 8) Step idx, 9) Plan thought flag, 10). Plan index
    plan_summarization_prompt = BasicThoughtPromptFormat(
        head="{}\nFor the given question, let's focus on summarize the plan that underpins the reasoning step {}.\n",
        content="\n{}\n{}\n\n{}\n\n",
        target="Please review the reasoning steps within {} and their corresponding plans within {} and proceed to summarize the plan of Step {} within {}, i.e., Plan {}.",
        notice=" Only direct output summarized plan. Do not include the Plan index in the output. Remember that the plan is a high-level, question-agnostic principle. Do not include any question or reasoning step content in the plan.",
        tail="",
        prompt="",
    )

    # Braces: 1) Question
    #   2) Generated Thought, 3) Plan Assessment
    #   3) Plan candidates flag
    assess_first_plan_prompt = BasicThoughtPromptFormat(
        head="{}\nFor the given question, let's focus on assessing whether the plan guides the generation of an effective first step.\n",
        content="{}\n{}\n\n",
        target="Please assess Plan {} within {}. Notice that the reasoning Step 1 within {} is guided by the Plan 1 within {}.",
        notice=" Only output the assessment score ranging from 0 to 1, while a higher score means a better plan as reasoning guidance.",
        tail="",
        prompt="",
    )

    # Braces: 1) Question
    #   3) Reasoning chain 5) Generated Thought 6) Plan Assessment
    #   7) Reasoning chain flag
    assess_next_plan_prompt = BasicThoughtPromptFormat(
        head="{}\nFor the given question, Let's focus on assessing whether the plan can guide the generation of an effective next reasoning step.\n",
        content="\n{}\n\n{}\n{}\n\n",
        target="Please review the reasoning steps already taken within the tag {} and the generated next Step {} within {} guided by the Plan {} within the tag {}, then assess this Plan {}.",
        notice=" Only output the assessment score ranging from 0 to 1, while a higher score means a better plan as reasoning guidance.",
        tail="",
        prompt="",
    )

    # Braces: 1) Question 2) Step index
    #   3) Plan chain 4) Plan pool
    #   5) Plan index, 6) Plan pool flag 7) Plan index
    compare_plan_prompt = BasicThoughtPromptFormat(
        head="Let's focus on whether the given plan exists in the plan pool.\n",
        content="\n{}\n\n{}\n\n",
        target="Please judge whether the Plan {} within the tag {} already exists in the plans within the tag {}.",
        notice="Only output True if exists, or False if not. Remember that plan is a high-level, question-agnostic principle. Do not focus on text details but on the logic, high-level ideas, theorems, or rules.",
        tail="",
        prompt="",
    )

    plan_generation_prompt = BasicThoughtPromptFormat(
        head="{}",
        content="{}",
        target="Please generate the Plan {} for guiding the next reasoning step {}.",
        notice="",
        tail="",
        prompt="",
    )


class ExplicitPlanSystemPrompts(BaseSystemPrompts):
    """A series of system prompts for the explicit plan operation."""

    plan_summarization_prompt: str = (
        """You are an expert in identifying, extracting, and summarizing the plan that underpins one reasoning step. The summarized plan is the generation instruction of the step and, thus, contains an explicit and precise principle of how to generate this step. Please get such a plan from the given reasoning step. Start by reviewing the given question and any previous reasoning steps already taken, then briefly summarize the explicit plan of the given reasoning step without including many details."""
    )

    plan_generation_prompt: str = (
        """You are an expert reasoning planner who excels at generating specific plans to guide the generation of the next steps when using step-by-step reasoning to solve problems. A plan behaves as a detailed, precise and specific instruct of generating the next reasoning step. After reviewing the given question and existing previous reasoning steps, please generate the plan that can guide the generation of the next reasoning step."""
    )


class ExplicitPlanPrompts:
    """A series of prompts for the plan generation and creation."""

    step_plan_summary_prompt = """Let's briefly summarize the plan of Step {} and directly generate the plan presenting precise, explicit, and concise guidance to generate this step."""


# that does not have many details and is the
class PlanThoughtGenerationPrompts(ThoughtGenerationPrompts):
    """
    A base class to organize the prompt with the plan for the thought generation.
    """

    # Corresponding to the I_G^{prime} of the p-RAR paper
    #  Braces: 1) Question, 2) Reasoning chain, 3) Plan chain
    #   4) Reasoning chain flag, 5) Plan chain flag 6) Step index
    # next_step_prompt = BasicThoughtPromptFormat(
    #     head="{}Let's focus on carefully and directly generating the next possible reasoning step for the reasoning steps below.\n",
    #     content="\n{}\n\n{}\n\n",
    #     target="Please review the reasoning steps within the tag {} along with their plans within the tag {}, then proceed to directly generate the best next step, i.e., Step {}.",
    #     notice=" Only output the generated step. Do not include the Step index in the output.",
    #     tail="",
    #     prompt="",
    # )

    next_step_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on carefully and directly generating the next possible reasoning step for the reasoning steps below.\n",
        content="\n{}\n\n",
        target="Please review the reasoning steps within the tag {}, then proceed to directly generate the best next step, i.e., Step {}.",
        notice=" Only output the generated step. Do not include the Step index in the output.",
        tail="",
        prompt="",
    )

    # Corresponding to the I_G of the p-RAR paper
    # Braces: 1) Question, 2) Plan, 3) Plan flag
    plan_guide_first_step_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on following the plan to directly generating the first reasoning step.\n",
        content="\n{}\n\n",
        target="Please follow the Plan 1 provided within the tag {} to generate a well-crafted first step, i.e., Step 1.",
        notice=" Only output the generated step. Do not include the Step index in the output.",
        tail="",
        prompt="",
    )
    # Braces: 1). Question,
    # 2) Reasoning chain, 3) Plan chain 4) Plan
    # 5) Reasoning chain flag, 6) Plan chain flag, 6) Plan index,
    # 7) Plan flag 8) Step index
    plan_guide_next_step_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on following the plan to directly generate the next reasoning step for the reasoning steps below.\n",
        content="\n{}\n{}\n\n{}\n\n",
        target="Please review the reasoning steps within the tag {} along with their plans within the tag {}, then follow the Plan {} within the tag {} to proceed to directly generate the best next step, i.e., Step {}.",
        notice=" Only output the generated step. Do not include the Step index in the output.",
        tail="",
        prompt="",
    )

    # Corresponding to the I_E of the p-RAR paper
    # Braces: 1) Question
    #   2). Plan exclusion
    #   4) Plan exclusion flag
    exclusive_plan_first_step_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on avoiding repeating the given plans to directly generate the first reasoning step to start addressing the question.\n",
        content="\n{}\n\n",
        target="As the start of reasoning, please exclude Plan 1 listed within the tag {} to generate a well-crafted first step, i.e., Step 1.",
        notice=" Only output the generated step. Do not include the Step index in the output.",
        tail="",
        prompt="",
    )
    # Braces: 1) Question,
    #   2) Reasoning chain, 3) Plan chain 4) Plan Exclusion
    #   5) Reasoning chain flag, 6) Plan chain flag, 7) Plan candidate index
    #   8) Plan candidate flag 9) Step index
    exclusive_plan_next_step_prompt = BasicThoughtPromptFormat(
        head="{}Let's focus on avoiding using the given plans to carefully and directly generate the next possible reasoning step for the reasoning steps below.\n",
        content="\n{}\n{}\n\n{}\n\n",
        target="Please review the reasoning steps within the tag {} and their plans within the tag {}, then specifically avoid repeating all Plan {} listed within tag {} to proceed to directly generate the best next step, i.e., Step {}.",
        notice=" Only output the generated step. Do not include the Step index in the output.",
        tail="",
        prompt="",
    )


class BasePlanThoughtPrompts(BaseThoughtPrompts):
    """A base class to organize the plan-based thought prompts"""

    generation: PlanThoughtGenerationPrompts = PlanThoughtGenerationPrompts()
