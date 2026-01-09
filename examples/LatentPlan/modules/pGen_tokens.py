"""
Define the tokens that are used across the pGen project.
"""

## Ones used by the decoder of the concept learner
# Placeholder of the plan to be reconstructed
# This placeholder is not trainable and its embedding is
# directly replaced by the plan encodings from the encoder
PLAN_PLACEHOLDER = "<PLANPL>"
# Indicator added after the placeholder to indicate the reconstruction
# -> Trainable.
# The number of the latent plans here is the same as the
# concepts defined in the concept learner
LATENT_PLAN_IDX = "<IPR{}>"  # add the index when use


# The explicit plan presenting the plan related to the question
# More specific
BLOCK_PLAN_START = "<plans>"
BLOCK_PLAN_END = "</plans>"

# The explicit plan presenting the plan related to the question
# More specific
PLAN_START = "<plan>"
PLAN_END = "</plan>"

# The indicate of the reasoning steps
STEP_START = "<step>"
STEP_END = "</step>"

BLOCK_STEP_START = "<steps>"
BLOCK_STEP_END = "</steps>"

# There are two types of plan appearance in the reasoning
# Please adjust `plan_appear` to enable:
#   - none: the plans will not appear in the reasoning
#   - head: the plans will be placed as a whole in the head of the reasoning
#   - step: each plan will be placed in the front of each reasoning step

# For example, plan_appear == "head"
# """
#     <plans> ..... </plans>
#     <steps> ..... </steps>
# """


# For example, plan_appear == "step"
# """
#     <plan> ..... </plan>
#     <step> ..... </step>

#     <plan> ..... </plan>
#     <step> ..... </step>

#     ...
# """
