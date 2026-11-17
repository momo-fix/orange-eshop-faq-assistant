import random
import re
import numpy as np
import torch


def _fallback_yaml_load(text: str) -> dict:
    """Minimal nested YAML parser fallback if PyYAML is not installed."""
    root = {}
    stack = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, val = line.strip().partition(":")
        key = key.strip()
        val = val.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            new_dict = {}
            parent[key] = new_dict
            stack.append((indent, new_dict))
        else:
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            elif val.lower() in ("true", "false"):
                val = val.lower() == "true"
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
            parent[key] = val
    return root


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    try:
        import yaml
        return yaml.safe_load(content)
    except ImportError:
        return _fallback_yaml_load(content)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(text: str) -> str:
    text = str(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def build_prompt(question: str, cfg: dict) -> str:
    user_tok = cfg["chat_tokens"]["user"]
    asst_tok = cfg["chat_tokens"]["assistant"]
    return f"{user_tok}\n{question}\n{asst_tok}\n"


def load_model_and_tokenizer(model_path: str, cfg: dict, is_sft: bool = False):
    """Unified loader for base, +CPT, and +SFT models."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, cfg: dict, is_sft: bool = False):
    """Single shared generation function used across base, +CPT, and +SFT."""
    model.eval()
    max_new_tokens = cfg["eval"]["max_new_tokens"]
    do_sample = cfg["eval"]["do_sample"]

    if is_sft:
        prompt = build_prompt(question, cfg)
        end_id = tokenizer.convert_tokens_to_ids(cfg["chat_tokens"]["end"])
        eos_ids = [end_id, tokenizer.eos_token_id]
    else:
        prompt = f"Question: {question}\nAnswer:"
        eos_ids = tokenizer.eos_token_id

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    if isinstance(eos_ids, list):
        stopped = any(tok in new_ids for tok in eos_ids)
    else:
        stopped = eos_ids in new_ids
    return text, stopped
