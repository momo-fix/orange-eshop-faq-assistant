import argparse
import os
import time
import pandas as pd
from src.utils import generate_answer, load_config, load_model_and_tokenizer, set_global_seed

REFUSAL_PHRASE = "don’t have verified information"


def evaluate_model(label: str, model_path: str, test_df: pd.DataFrame, out_csv: str, cfg: dict, is_sft: bool):
    model, tokenizer = load_model_and_tokenizer(model_path, cfg, is_sft=is_sft)
    rows = []
    for _, row in test_df.iterrows():
        t0 = time.perf_counter()
        ans, stopped = generate_answer(model, tokenizer, row["question"], cfg, is_sft=is_sft)
        dt = round(time.perf_counter() - t0, 4)
        auto_ref = (REFUSAL_PHRASE in ans.lower()) or ("don't have verified information" in ans.lower())
        rows.append({
            "id": row["id"],
            "category": row["category"],
            "should_refuse": row.get("should_refuse", False),
            "question": row["question"],
            "expected_answer": row["expected_answer"],
            "model_answer": ans,
            "stopped_cleanly": stopped,
            "auto_detected_refusal": auto_ref,
            "latency_s": dt,
            "grader1_score": "",
            "grader2_score": "",
            "final_score": "",
            "error_cause": "",
            "note": "",
        })
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[{label}] Saved {len(rows)} predictions to {out_csv}")


def summarize_results(cfg: dict):
    files = [
        ("Small model, base", cfg["eval"]["base_results"]),
        ("+ continued pretraining", cfg["eval"]["cpt_results"]),
        ("+ SFT", cfg["eval"]["sft_results"]),
    ]
    print("\n=== PHASE 2 EVALUATION SUMMARY ===")
    print(f"{'Version':<26} | {'Accuracy':<16} | {'Refusal Acc':<16} | {'Avg Latency':<12}")
    print("-" * 78)
    for label, path in files:
        if not os.path.exists(path):
            print(f"{label:<26} | Missing file: {path}")
            continue
        df = pd.read_csv(path)
        scores = df["final_score"].fillna(df["grader1_score"])
        acc = scores.sum() / len(df) * 100.0
        ref_df = df[df["should_refuse"] == True]
        ref_ok = (scores[df["should_refuse"] == True] == 1.0).sum()
        ref_acc = (ref_ok / len(ref_df) * 100.0) if len(ref_df) else 0.0
        lat = df["latency_s"].mean()
        print(f"{label:<26} | {acc:5.1f}% ({scores.sum():4.1f}/30) | {ref_acc:5.1f}% ({ref_ok}/{len(ref_df)})   | {lat:.3f}s")


def main():
    parser = argparse.ArgumentParser(description="Run and summarize Phase 2 model evaluations.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run inference on base, +CPT, and +SFT checkpoints (requires local checkpoints/).",
    )
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    set_global_seed(cfg["seed"])

    if args.run:
        test_df = pd.read_csv(cfg["eval"]["test_file"])
        # Preserve should_refuse flags from existing graded results if present
        if os.path.exists(cfg["eval"]["sft_results"]):
            ref_map = pd.read_csv(cfg["eval"]["sft_results"]).set_index("id")["should_refuse"]
            test_df["should_refuse"] = test_df["id"].map(ref_map).fillna(False)

        evaluate_model("Base", cfg["model_name"], test_df, cfg["eval"]["base_results"], cfg, is_sft=False)
        evaluate_model("+CPT", cfg["cpt"]["final_dir"], test_df, cfg["eval"]["cpt_results"], cfg, is_sft=False)
        evaluate_model("+SFT", cfg["sft"]["final_dir"], test_df, cfg["eval"]["sft_results"], cfg, is_sft=True)

    summarize_results(cfg)


if __name__ == "__main__":
    main()
