import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import transformers
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from src.utils import build_prompt, load_config, set_global_seed


def main():
    cfg = load_config("config.yaml")
    seed = cfg["seed"]
    set_global_seed(seed)
    transformers.set_seed(seed)

    cpt_dir = cfg["cpt"]["final_dir"]
    sft_cfg = cfg["sft"]
    max_length = cfg["max_length"]

    user_tok = cfg["chat_tokens"]["user"]
    asst_tok = cfg["chat_tokens"]["assistant"]
    end_tok = cfg["chat_tokens"]["end"]

    tokenizer = GPT2Tokenizer.from_pretrained(cpt_dir)
    tokenizer.pad_token = tokenizer.eos_token
    added = tokenizer.add_special_tokens({"additional_special_tokens": [user_tok, asst_tok, end_tok]})
    assert added == 3, f"Expected 3 added tokens, got {added}"

    model = GPT2LMHeadModel.from_pretrained(cpt_dir)
    model.resize_token_embeddings(len(tokenizer))
    end_id = tokenizer.convert_tokens_to_ids(end_tok)

    # Initialize <|end|> embedding from <|endoftext|> so the model stops cleanly
    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        emb[end_id] = emb[tokenizer.eos_token_id].clone()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def encode_example(question: str, answer: str) -> dict:
        prompt_ids = tokenizer(build_prompt(question, cfg))["input_ids"]
        answer_ids = tokenizer(answer)["input_ids"] + [end_id]
        ids = prompt_ids + answer_ids
        assert len(ids) <= max_length, f"Example exceeds max_length ({len(ids)} > {max_length})"
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prompt_ids) + answer_ids,
        }

    def sft_collator(batch):
        longest = max(len(b["input_ids"]) for b in batch)
        pad = tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor([b["input_ids"] + [pad] * (longest - len(b["input_ids"])) for b in batch]),
            "attention_mask": torch.tensor([b["attention_mask"] + [0] * (longest - len(b["attention_mask"])) for b in batch]),
            "labels": torch.tensor([b["labels"] + [-100] * (longest - len(b["labels"])) for b in batch]),
        }

    sft_df = pd.read_json(sft_cfg["pairs_file"], lines=True, dtype=False)
    train_rows = sft_df[sft_df["split"] == "train"]
    val_rows = sft_df[sft_df["split"] == "val"]

    train_ds = Dataset.from_list([encode_example(q, a) for q, a in zip(train_rows["question"], train_rows["answer"])])
    val_ds = Dataset.from_list([encode_example(q, a) for q, a in zip(val_rows["question"], val_rows["answer"])])

    sft_args = TrainingArguments(
        output_dir=sft_cfg["output_dir"],
        num_train_epochs=sft_cfg["epochs"],
        learning_rate=float(sft_cfg["learning_rate"]),
        per_device_train_batch_size=sft_cfg["batch_size"],
        per_device_eval_batch_size=sft_cfg["batch_size"],
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=sft_collator,
    )

    val_loss_before = trainer.evaluate()["eval_loss"]
    print(f"Validation loss before SFT: {val_loss_before:.3f}")
    trainer.train()

    os.makedirs(os.path.dirname(sft_cfg["loss_curve"]), exist_ok=True)
    log = trainer.state.log_history
    sft_train_loss = [(x["epoch"], x["loss"]) for x in log if "loss" in x]
    sft_eval_loss = [(0, val_loss_before)] + [(x["epoch"], x["eval_loss"]) for x in log if "eval_loss" in x]
    plt.figure()
    plt.plot(*zip(*sft_train_loss), marker="o", label="train loss")
    plt.plot(*zip(*sft_eval_loss), marker="o", label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Supervised Fine-Tuning (SFT) Loss Curve")
    plt.legend()
    plt.savefig(sft_cfg["loss_curve"], bbox_inches="tight")
    plt.close()

    os.makedirs(sft_cfg["final_dir"], exist_ok=True)
    trainer.save_model(sft_cfg["final_dir"])
    tokenizer.save_pretrained(sft_cfg["final_dir"])
    print(f"Saved SFT checkpoint to {sft_cfg['final_dir']}")


if __name__ == "__main__":
    main()
