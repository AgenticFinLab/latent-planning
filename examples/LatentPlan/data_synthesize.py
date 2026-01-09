"""
Synthesize the dataset to be uploaded to the huggingface.
"""

from synthetic.synthetic_pipeline import PlanSyntheticPipeline


def _main():

    # Synthesize the trainset
    # Mask train part when the dataset does not have a train split.
    # train_pipeline = PlanSyntheticPipeline(phase="train")
    # train_pipeline.initialize()
    # train_pipeline.synthesize()

    test_pipeline = PlanSyntheticPipeline(phase="test")
    test_pipeline.initialize()
    test_pipeline.synthesize()


if __name__ == "__main__":

    _main()
