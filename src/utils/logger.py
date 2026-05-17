import wandb
from typing import Any

class Logger:
    def __init__(self, project: str, name: str):
        self.run = wandb.init(
            entity="ltao02845-sun-yat-sen-university",
            project=project,
            name=name,
        )
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        # everything that starts with train/ is tied to train_step
        wandb.define_metric("train/*", step_metric="train_step")
        # everything that starts with eval/ is tied to eval_step
        wandb.define_metric("eval/*", step_metric="eval_step")
        self.train_step = 0
        self.eval_step = 0

    def log_train(self, item_dict: dict[str, Any]):
        self.train_step += 1
        log_dict = {"train_step": self.train_step}
        for name,value in item_dict.items():
            log_dict[f"train/{name}"] = value
        self.run.log(log_dict)

    def log_eval(self, item_dict: dict[str, Any]):
        self.eval_step += 1
        log_dict = {"eval_step": self.eval_step}
        for name,value in item_dict.items():
            log_dict[f"eval/{name}"] = value
        self.run.log(log_dict)

    def finish(self):
        self.run.finish()