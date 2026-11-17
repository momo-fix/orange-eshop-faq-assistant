import math
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)
from src.utils import load_config, set_global_seed


def make_blocks(texts, tokenizer, block_size: int) -> Dataset:
    all_ids = []
    for t in texts:
        all_ids += tokenizer(str(t))["input_ids"] + [tokenizer.eos_token_id]
    blocks = [all_ids[i : i + block_size] for i in range(0, len(all_ids), block_size)]
    return Dataset.from_dict({"input_ids": blocks})


def compute_perplexity(model, dataset: Dataset) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for ids in dataset["input_ids"]:
        x = torch.tensor([ids]).to(model.device)
        with torch.no_grad():
            loss = model(x, labels=x).loss.item()
        n = len(ids) - 1
        total_loss += loss * n
        total_tokens += n
    return math.exp(total_loss / total_tokens)


def main():
    cfg = load_config("config.yaml")
    seed = cfg["seed"]
    set_global_seed(seed)

    model_name = cfg["model_name"]
    block_size = cfg["max_length"]
    cpt_cfg = cfg["cpt"]

    train_df = pd.read_csv(cpt_cfg["train_file"])
    val_df = pd.read_csv(cpt_cfg["val_file"])

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_dataset = make_blocks(train_df["text"].tolist(), tokenizer, block_size)
    val_dataset = make_blocks(val_df["text"].tolist(), tokenizer, block_size)

    base_ppl = compute_perplexity(model, val_dataset)
    print(f"Base validation perplexity: {base_ppl:.2f}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=cpt_cfg["output_dir"],
        num_train_epochs=cpt_cfg["epochs"],
        learning_rate=float(cpt_cfg["learning_rate"]),
        per_device_train_batch_size=cpt_cfg["batch_size"],
        per_device_eval_batch_size=cpt_cfg["batch_size"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    cpt_ppl = compute_perplexity(trainer.model, val_dataset)
    print(f"CPT validation perplexity : {cpt_ppl:.2f}")

    os.makedirs(os.path.dirname(cpt_cfg["loss_curve"]), exist_ok=True)
    log = trainer.state.log_history
    train_loss = [(x["epoch"], x["loss"]) for x in log if "loss" in x]
    eval_loss = [(x["epoch"], x["eval_loss"]) for x in log if "eval_loss" in x]
    plt.figure()
    plt.plot(*zip(*train_loss), marker="o", label="train loss")
    plt.plot(*zip(*eval_loss), marker="o", label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Continued Pretraining (CPT) Loss Curve")
    plt.legend()
    plt.savefig(cpt_cfg["loss_curve"], bbox_inches="tight")
    plt.close()

    os.makedirs(cpt_cfg["final_dir"], exist_ok=True)
    trainer.save_model(cpt_cfg["final_dir"])
    tokenizer.save_pretrained(cpt_cfg["final_dir"])
    print(f"Saved CPT checkpoint to {cpt_cfg['final_dir']}")


if __name__ == "__main__":
    main()
