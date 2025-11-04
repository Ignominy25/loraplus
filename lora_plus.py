from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from peft.tuners import lora
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import (EvalPrediction, PreTrainedModel,
                                  PreTrainedTokenizerBase, TrainerCallback)
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import is_sagemaker_mp_enabled, logging

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

logger = logging.get_logger(__name__)


@dataclass
class LoraPlusTrainingArguments(TrainingArguments):
    loraplus_lr_ratio: Optional[float] = field(
        default=None, metadata={"help": "loraplus learning rate ratio lr_B / lr_A."}
    )
    loraplus_lr_embedding: Optional[float] = field(
        default=1e-6,
        metadata={"help": "loraplus learning rate for lora embedding layers."},
    )
    lr_step_size: int = field(
        default=100,
        metadata={"help": "Number of steps after which the learning rate is updated."},
    )
    lr_gamma: float = field(
        default=0.1,
        metadata={"help": "Factor by which the learning rate is multiplied at each step."},
    )
    predict_with_generate: bool = field(
        default=False,
        metadata={"help": "Whether to use generate() for predictions (for seq2seq models)."},
    )
    generation_max_length: int = field(
        default=128,
        metadata={"help": "Maximum length for generation."},
    )
    generation_num_beams: int = field(
        default=4,
        metadata={"help": "Number of beams for generation."},
    )
    generation_length_penalty: float = field(
        default=2.0,
        metadata={"help": "Length penalty for generation. Values > 1.0 encourage longer sequences."},
    )
    generation_early_stopping: bool = field(
        default=True,
        metadata={"help": "Whether to stop beam search when at least num_beams sentences are finished."},
    )


def get_module(name, opt_model):
    """
    Retrieve a module from a model using its parameter name.
    Args:
        name (str): Full name of the parameter, typically including module path.
        opt_model (torch.nn.Module): The model from which to retrieve the module.

    Returns:
        Module corresponding to the given name.
    """
    parent_idx = 2 if "lora" in name else 1
    module_names = name.split(sep=".")[:-parent_idx]
    module = reduce(getattr, module_names, opt_model)
    return module


def create_loraplus_optimizer(
    opt_model,
    optimizer_cls,
    optimizer_kwargs,
    loraplus_lr_ratio,
    loraplus_lr_embedding=None,
):
    """
    Creates an optimizer for the given model, applying LoRA-specific learning rate adjustments to different parameter groups.

    Args:
        opt_model (torch.nn.Module): The model for which the optimizer is being created.
        optimizer_cls (class): The class of the optimizer to be used (e.g., torch.optim.Adam).
        optimizer_kwargs (dict): A dictionary of keyword arguments for the optimizer's initialization.
        loraplus_lr_ratio (float): The learning rate ratio to be applied to LoRA parameters.
        loraplus_lr_embedding (float, optional): A specific learning rate for embedding parameters, with a default value if not provided.

    Returns:
        An instance of the specified optimizer class configured with the model's parameters organized into groups with custom learning rates.
    """

    assert loraplus_lr_ratio is not None, "loraplus_lr_ratio must be provided."

    if loraplus_lr_embedding is None:
        loraplus_lr_embedding = 1e-6

    decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    param_groups = {
        "groupA": {},
        "groupB": {},
        "groupB_no_decay": {},
        "embedding": {},
    }

    for name, param in opt_model.named_parameters():
        if not param.requires_grad:
            continue

        module = get_module(name, opt_model)
        if isinstance(module, lora.Embedding):
            param_groups["embedding"][name] = param
        elif "lora_B" in name or param.ndim == 1:
            if name in decay_parameters:
                param_groups["groupB"][name] = param
            else:
                param_groups["groupB_no_decay"][name] = param
        else:
            param_groups["groupA"][name] = param

    assigned_param_groups = ""
    for group in param_groups:
        assigned_param_groups += f"{group}\n {list(param_groups[group].keys())}\n\n"
    logger.info(assigned_param_groups)

    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["groupA"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": list(param_groups["embedding"].values()),
            "weight_decay": weight_decay,
            "lr": loraplus_lr_embedding,
        },
        {
            "params": list(param_groups["groupB"].values()),
            "weight_decay": weight_decay,
            "lr": lr * loraplus_lr_ratio,
        },
        {
            "params": list(param_groups["groupB_no_decay"].values()),
            "weight_decay": 0.0,
            "lr": lr * loraplus_lr_ratio,
        },
    ]

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in opt_model.modules():
            if isinstance(module, nn.Embedding):
                skipped += sum(
                    {p.data_ptr(): p.numel() for p in module.parameters()}.values()
                )
                logger.info(f"skipped {module}: {skipped/2**20}M params")
                manager.register_module_override(module, "weight", {"optim_bits": 32})
                logger.debug(f"bitsandbytes: will optimize {module} in fp32")
        logger.info(f"skipped: {skipped/2**20}M params")

    return optimizer


class CustomLRScheduler:
    def __init__(self, optimizer, lr_step_size, lr_gamma, loraplus_lr_ratio):
        self.optimizer = optimizer
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count % self.lr_step_size == 0:
            for param_group in self.optimizer.param_groups:
                # Apply lr_gamma only to groupB and groupB_no_decay
                if param_group.get("lr") and param_group["lr"] not in [
                    self.optimizer.defaults.get("lr"),  # groupA
                    param_group.get("loraplus_lr_embedding"),  # embedding
                ]:
                    param_group["lr"] *= self.lr_gamma

        # Log the current ratio of learning rates for groupA and groupB
        groupA_lr = None
        groupB_lr = None
        for param_group in self.optimizer.param_groups:
            if param_group.get("lr") == self.optimizer.defaults.get("lr"):
                groupA_lr = param_group["lr"]
            elif param_group.get("lr") and param_group["lr"] != self.optimizer.defaults.get("lr"):
                groupB_lr = param_group["lr"]

        if groupA_lr is not None and groupB_lr is not None and (self.step_count % self.lr_step_size == 0):
            current_ratio = groupB_lr / groupA_lr
            print(f"\n[DEBUG] Step {self.step_count}: Current Ratio (groupB/groupA): {current_ratio}")

    def get_last_lr(self):
        """
        Returns the current learning rates for all parameter groups.
        """
        return [param_group["lr"] for param_group in self.optimizer.param_groups]


class LoraPlusTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: LoraPlusTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ):
        assert isinstance(
            args, LoraPlusTrainingArguments
        ), "args must be of type LoraPlusTrainingArguments"
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

    def create_optimizer(self):
        """
        Overrides the method to create an optimizer with LoRA+ specific adjustments and integrates a custom lr_ratio scheduler.
        """
        if self.args.loraplus_lr_ratio is None:
            return super().create_optimizer()

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            loraplus_lr_ratio = getattr(self.args, "loraplus_lr_ratio", None)
            loraplus_lr_embedding = getattr(self.args, "loraplus_lr_embedding", None)
            self.optimizer = create_loraplus_optimizer(
                opt_model,
                optimizer_cls,
                optimizer_kwargs,
                loraplus_lr_ratio,
                loraplus_lr_embedding,
            )

            # Integrate custom lr_ratio scheduler
            self.lr_scheduler = CustomLRScheduler(
                self.optimizer,
                lr_step_size=self.args.lr_step_size,
                lr_gamma=self.args.lr_gamma,
                loraplus_lr_ratio=self.args.loraplus_lr_ratio,
            )

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def training_step(self, *args, **kwargs):
        """
        Overrides the training step to update the custom lr_ratio scheduler.
        """
        # Perform the standard training step
        output = super().training_step(*args, **kwargs)

        # Update the custom lr_ratio scheduler
        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

        return output
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        Override prediction_step to support generation for seq2seq models.
        """
        # Check if model has generate method (seq2seq models like T5, BART)
        if hasattr(model, "generate") and self.args.predict_with_generate:
            has_labels = "labels" in inputs
            inputs = self._prepare_inputs(inputs)
            
            # Generate predictions
            gen_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_length": self.args.generation_max_length if hasattr(self.args, 'generation_max_length') else 128,
                "num_beams": self.args.generation_num_beams if hasattr(self.args, 'generation_num_beams') else 4,
            }
            
            generated_tokens = model.generate(**gen_kwargs)
            
            # Compute loss if labels are present
            if has_labels:
                with torch.no_grad():
                    outputs = model(**inputs)
                    loss = outputs.loss
            else:
                loss = None
            
            labels = inputs.get("labels")
            
            # Keep as tensors for the Trainer's evaluation loop
            # They will be converted to numpy later in compute_metrics
            return (loss, generated_tokens, labels)
        else:
            # Fall back to default behavior for classification
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
