import os
import hydra
import torch
import lightning as L
import wandb
from omegaconf import DictConfig
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
from plora import PLoraConfig, PLoraModel

from data_utils import PhishingDataModule, IDGDataModule
from transformer import Classifier, ShallowTransformer

dataset_info = {
    "phishing": (PhishingDataModule,"./dataset/phishing/phishing-Emd"),
    "idg": (IDGDataModule, "./dataset/IDG/IDG-Emd")
}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set up
    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")
    exp_dir = "~/scratch/ilpimm"

    # Load the data
    data_module, data_dir = dataset_info[cfg.dataset]
    dm = data_module(
        batch_size=cfg.trainer.batch_size,
        data_dir=data_dir
    )
    dm.setup("fit")

    # Common checkpoint & logger setup
    def get_callbacks_and_logger(phase):
        checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            dirpath=os.path.join(exp_dir, f"checkpoints_{phase}"),
            filename=f"epoch_{phase}_{{epoch:02d}}-val_loss_{{val/loss:.2f}}",
            save_top_k=cfg.trainer.save_top_k,
            auto_insert_metric_name=False,
        )

        if cfg.wandb_logger:
            logger = loggers.WandbLogger(
                name=f"{os.path.basename(exp_dir)}_{phase}",
                project="phishing",
                save_dir=exp_dir,
                config=dict(cfg),
                notes=cfg.run_notes,
            )
        else:
            logger = loggers.TensorBoardLogger(exp_dir, name=phase)

        return checkpoint_callback, logger

    # Phase 1: Base model training
    if cfg.model_checkpoint:
        print("Loading pretrained base model from checkpoint")
        base_model = ShallowTransformer(
            num_classes=dm.num_classes,
            emb_size=dm.emb_size,
            seq_length=cfg.model.seq_length,
            dropout=cfg.model.dropout,
        )
        base_model.config = {k: v if not isinstance(v, DictConfig) else dict(v) for k, v in dict(cfg.model).items()}
        best_base_model = Classifier.load_from_checkpoint(
            cfg.model_checkpoint,
            model=base_model,
            lr=cfg.trainer.lr
        )
        checkpoint_callback = None
    else:
        print("Starting Phase 1: Base Model Training")
        base_model = ShallowTransformer(
            num_classes=dm.num_classes,
            emb_size=dm.emb_size,
            seq_length=cfg.model.seq_length,
            dropout=cfg.model.dropout,
        )
        base_model.config = {k: v if not isinstance(v, DictConfig) else dict(v) for k, v in dict(cfg.model).items()}
        model = Classifier(
            base_model,
            lr=cfg.trainer.lr,
            plora_train=False
        )
        checkpoint_callback, logger = get_callbacks_and_logger("base")
        trainer = L.Trainer(
            max_epochs=cfg.trainer.max_epochs,
            check_val_every_n_epoch=2,
            logger=logger,
            callbacks=[checkpoint_callback],
            devices=cfg.trainer.devices,
        )
        model = torch.compile(model)
        trainer.fit(model, datamodule=dm)
        dm.setup("test")
        base_results = trainer.test(ckpt_path="best", datamodule=dm)
        wandb.finish()
        best_base_model = Classifier.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            model=base_model,
            lr=cfg.trainer.lr
        )
    # Phase 2: P-LoRA fine-tuning
    print("Starting Phase 2: P-LoRA Fine-tuning")

    # Configure P-LoRA
    plora_config = PLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias=cfg.bias,
        num_virtual_users=dm.n_users,
        user_token_dim=cfg.user_token_dim,
    )

    plora_model = PLoraModel(
        model=best_base_model.model,  # Use the trained base model
        config={"default": plora_config},
        adapter_name="default"
    )

    # Create new classifier with P-LoRA
    plora_classifier = Classifier(
        plora_model,
        lr=cfg.trainer.lr,
        plora_train=True
    )

    # New trainer for P-LoRA phase
    checkpoint_callback, logger = get_callbacks_and_logger("plora")

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        check_val_every_n_epoch=2,
        logger=logger,
        callbacks=[checkpoint_callback],
        devices=cfg.trainer.devices,
    )
    plora_classifier = torch.compile(plora_classifier)
    trainer.fit(plora_classifier, datamodule=dm)

    # Test P-LoRA model
    plora_results = trainer.test(ckpt_path="best", datamodule=dm)

    # Print comparison
    print("Base Model Results:", base_results)
    print("P-LoRA Model Results:", plora_results)

if __name__ == "__main__":
    main()