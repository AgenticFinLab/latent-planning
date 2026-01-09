"""
Implement the init project to create the project.
"""

from projinit.platform_init import ProjectInfo


class pGenProjectInfo(ProjectInfo):
    """The project info for the pGen."""

    def __init__(self, project_type: str):
        super().__init__()

        assert project_type in ["learner", "reasoner"]

        self.project_type = project_type

        self.plan_type = self.train_config["plan_type"]
        self.plan_status = self.train_config["reason_plan_space"]
        self.plan_appear = self.train_config["plan_appear"]

        data_synthetic_config = self.train_config["data_synthetic"]
        self.data_folder = data_synthetic_config["synthesized_path"]
        self.synthesized_data_name = data_synthetic_config["synthesized_name"]
        self.train_dataname = None
        self.test_dataname = None

        if self.project_type == "learner":
            if self.plan_type == "skeleton":
                self.train_dataname = data_synthetic_config["ft_train_dataname"]
                self.test_dataname = data_synthetic_config["ft_test_dataname"]
            elif self.plan_type == "explicit":
                self.train_dataname = data_synthetic_config["ft_train_e_dataname"]
                self.test_dataname = data_synthetic_config["ft_test_e_dataname"]
            else:
                raise ValueError(
                    f"The plan type {self.plan_type} does not exist in skeleton|explicit."
                )
        elif self.project_type == "reasoner":
            if self.plan_type == "skeleton":
                self.train_dataname = data_synthetic_config[
                    "ft_train_plan_reason_dataname"
                ]
                self.test_dataname = data_synthetic_config[
                    "ft_test_plan_reason_dataname"
                ]
            elif self.plan_type == "explicit":
                self.train_dataname = data_synthetic_config[
                    "ft_train_e_plan_reason_dataname"
                ]
                self.test_dataname = data_synthetic_config[
                    "ft_test_e_plan_reason_dataname"
                ]
            else:
                raise ValueError(
                    f"The plan type {self.plan_type} does not exist in skeleton|explicit."
                )
