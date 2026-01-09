"""
Prompts related to the plan-based reasoning.
"""


class PlanSystemPrompts:
    """A series of system prompts for the plan operation."""

    answer_decompose_prompt = """You are an expert in decomposing an answer into a sequence of logical reasoning steps without altering its content in any way. Carefully read the given question and its answer, then break the answer into distinct reasoning steps. Do not add, remove, or change any words or characters from the original answer.
    Decompose according to the natural flow of reasoning: group closely connected lines together when appropriate, but separate major logical moves into distinct steps.
    Instructions:
    - Decompose the answer into individual steps based on its logical structure.
    - Do not over-split minor operations that belong naturally together.
    - Preserve the exact wording and punctuation of the original answer.
    - Only insert a step identifier (e.g., "Step 1:", "Step 2:") at appropriate points, without modifying the text itself.

    Goal: Make the reasoning process clear and explicit at a natural level of detail, without fragmenting the answer unnecessarily and without altering the answer content in any way."""

    explicit_plan_summary_prompt: str = (
        """You are an expert at identifying and summarizing the explicit plans that guide reasoning steps. Your task is to extract one specific plan for each reasoning step. Each plan should be expressed as a specialized reasoning instruction — a concrete, task-specific principle that directly relates to the question and the reasoning process.
        Instructions:  
        - Carefully review the question and the full sequence of reasoning steps.  
        - For each reasoning step, summarize only the specific plan that guides it.  
        - Each plan should capture concrete strategies, ideas, principles, or theorems relevant to the reasoning.  
        - Insert a plan identifier before each summarized plan, i.e. "Plan 1:", "Plan 2:", and so on, in the order of the reasoning steps.

        Goal: Produce a list of summarized reasoning plans — one per step — that accurately reflect the structure and intent behind each step in the reasoning process."""
    )

    single_skeleton_plan_summary_prompt: str = (
        """You are an expert in identifying, extracting, and summarizing the plan that underlies one specific reasoning step. The summarized plan should be a general-purpose reasoning instruction — a high-level, question-agnostic principle. 
        Start by reviewing the given question, any previous reasoning steps with their associated plans, and the specific reasoning step in focus. Then, extract and summarize the plan that guides this particular step.
        Instructions:
        - Identify the highest-level idea, principle, rule, or theorem that informs the step.
        - Express the plan in abstract, general terms, without referring to the specific content of the question or step.
        - Keep the summary brief, clear, and focused on the guiding reasoning pattern.

        Goal: Capture the general reasoning strategy behind this specific step. """
    )

    skeleton_plan_summary_prompt: str = (
        """You are an expert at identifying and summarizing the underlying skeleton plans behind reasoning steps. Your task is to extract one high-level plan per reasoning step. Each plan should be expressed as a general-purpose reasoning instruction — a broad, question-agnostic principle that guides reasoning.
        Instructions:
        - Carefully review the question and the full chain of reasoning steps.
        - For each reasoning step, summarize only the underlying plan that guides it.
        - Each plan should capture general strategies, ideas, principles, or theorems.
        - Do not include any specific details from the question and reasoning steps.
        - Insert a plan identifier before each summarized plan, i.e. "Plan 1:", "Plan 2:", and so on, matching the order of the reasoning steps.

        Goal: Produce a list of summarized reasoning plans, one per step, that reflect the deep structure of how reasoning is guided."""
    )


class PlanPrompts:
    """A series of prompts for the plan generation and creation."""

    answer_decompose_prompt = """Please decompose the given answer into logical reasoning steps without adding, removing, or changing any characters or words.
    """

    explicit_plan_summary_prompt = """"Please directly summarize the explicit plan that guides each reasoning step in the 'Steps:' above. For every step, provide a brief and precise principle that is specific to the task, closely related to the question, and facilitates the reasoning. This principle should serve as the instruction for generating the corresponding reasoning step. Insert a plan identifier before each summarized plan, i.e. "Plan 1:", "Plan 2:", and so on, following the order of the reasoning steps."""

    step_plan_summary_prompt = """Let's summarize the plan of the Step {} and directly generate the plan, which is a brief, high-level, question-agnostic principle, without including any question or reasoning step content."""

    skeleton_plan_summary_prompt = """Please directly summarize the skeleton plan that guides each reasoning step in the 'Steps:' above. For every step, provide a brief, high-level principle that is general-purpose and question-agnostic. Do not include any specific details from the question or the reasoning steps themselves. Insert a plan identifier before each summarized plan, i.e. "Plan 1:", "Plan 2:", and so on, matching the order of the reasoning steps."""
