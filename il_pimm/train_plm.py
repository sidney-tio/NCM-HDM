import os
import hydra
import torch
import wandb
from omegaconf import DictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
    default_data_collator
)

from peft import PeftModel, PeftConfig
from plora import PLoraConfig, PLoraModel, PMemoryLoraModel, PLoRaWrapper
from plm_dataset import make_supervised_data_module, smart_tokenizer_and_embedding_resize

dataset_info = {
    "phishing": "./dataset/phishing/phishing-Text/",
    "idg": "./dataset/IDG/IDG-Text/"
}

@hydra.main(version_base=None, config_path="conf_lm", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set up
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    exp_dir = "~/scratch/tlpimm"
    os.makedirs(exp_dir, exist_ok=True)

    # Initialize wandb if enabled
    if cfg.wandb_logger:
        wandb.init(
            name=f"{os.path.basename(exp_dir)}_llama_{cfg.dataset}",
            project="phishing",
            dir=exp_dir,
            config=dict(cfg),
            notes=cfg.run_notes,
        )

    model_load_params = {
        "device_map": cfg.trainer.devices,
        "torch_dtype": torch.float16 if cfg.trainer.fp16 else torch.float32
    }

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_checkpoint)
    print(f"Loading pretrained base model from checkpoint: {cfg.model_checkpoint}")
    peft_config = PeftConfig.from_pretrained(cfg.model_checkpoint)
    base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, **model_load_params)
    smart_tokenizer_and_embedding_resize(
                                        tokenizer=tokenizer,
                                        model=base_model,
                                        )
    base_model = PeftModel.from_pretrained(base_model, cfg.model_checkpoint)
    base_model = base_model.merge_and_unload()

    data_dir = dataset_info[cfg.dataset]

    data_module, n_users = make_supervised_data_module(tokenizer=tokenizer,data_args=data_dir, debug=cfg.debug)

    plora_config = PLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias=cfg.bias,
        num_virtual_users=n_users,
        user_token_dim=cfg.user_token_dim,
    )

    if cfg.input_type == 'seq':
        plora_model = PLoraModel(
            model=base_model,
            config={"default": plora_config},
            adapter_name="default"
        )
    elif cfg.input_type == "instance":
        plora_model = PMemoryLoraModel(
            model=base_model,
            config={"default": plora_config},
            adapter_name="default"
        )

    plora_model = PLoRaWrapper(plora_model)
    training_args = TrainingArguments(
        output_dir=os.path.join(exp_dir, f"{cfg.dataset}_plora_checkpoints"),
        num_train_epochs=cfg.trainer.max_epochs,
        per_device_train_batch_size=cfg.trainer.batch_size,
        per_device_eval_batch_size=cfg.trainer.batch_size,
        learning_rate=cfg.trainer.lr,
        weight_decay=0.01,
        logging_dir=os.path.join(exp_dir, "logs"),
        logging_steps=100,
        save_steps=200,
        save_strategy="steps",
        report_to="wandb" if cfg.wandb_logger else "none",
        ddp_find_unused_parameters=False,
    )

    # If multiple devices/processes are requested
    if torch.cuda.device_count() > 1:
        training_args.distributed_training = True
        training_args.ddp_backend = "nccl"  # For GPU
        training_args.deepspeed = None  # Could configure deepspeed here if needed

    # Set up the trainer
    trainer = Trainer(
        model=plora_model,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"]
    )

    # Train the model
    print("Starting P-LoRA Fine-tuning")
    trainer.train()

    # Evaluate on test set
    if cfg.test:
        print("Evaluating on test set")
        test_results = trainer.evaluate()
        print("P-LoRA Model Test Results:", test_results)

    # Save the final model
    # trainer.save_model(os.path.join(exp_dir, f"{cfg.dataset}_final_plora_model"))

    # Finish wandb run if it was used
    if cfg.wandb_logger:
        wandb.finish()


if __name__ == "__main__":
    main()

