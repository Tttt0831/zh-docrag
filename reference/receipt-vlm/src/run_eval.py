"""
真实评估脚本

加载训练好的 checkpoint，对测试集逐张做**真实推理**（model.generate），
将预测与真值统一归一化后，计算 src/eval.py 中的全套指标。

与旧版的区别：旧版直接把 target 复制加噪声当预测（从未调用模型）；
本版完全基于模型真实输出。

用法：
    python -m src.run_eval --checkpoint checkpoints/stage_a/best_model.pt \
        --data data/synthetic/train.jsonl --max-samples 20
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.infer import ReceiptInference, DEFAULT_PROMPT
from src.data.dataset import ReceiptDataset
from src.eval import compute_all_metrics, format_metrics_as_markdown
from src.utils.normalize import normalize_invoice_record


def evaluate(checkpoint, data_path, max_samples, dataset_name, device,
             max_new_tokens, temperature, output_dir, tokenizer_name=None):
    inferencer = ReceiptInference(checkpoint_path=checkpoint, device=device,
                                  tokenizer_name=tokenizer_name)

    dataset = ReceiptDataset(str(data_path), max_samples=max_samples, split="test")
    if len(dataset) == 0:
        raise SystemExit(f"测试集为空: {data_path}")

    print(f"\n对 {len(dataset)} 个样本做真实推理评估...")
    predictions, targets = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        pred = inferencer.generate_json(
            sample["image"],
            sample.get("prompt") or DEFAULT_PROMPT,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        # 归一化：保证预测与真值在同一口径下比较
        predictions.append(normalize_invoice_record(pred if isinstance(pred, dict) else {}))
        targets.append(normalize_invoice_record(sample["target_json"]))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(dataset)}")

    # 保存逐条预测：指标口径若要修正，可离线重算而不必重跑推理。
    # （F1 的计算口径就出过一次错，当时因为没存预测，四个规模全部要重跑。）
    preds_path = Path(output_dir)
    preds_path = preds_path if preds_path.is_absolute() else REPO_ROOT / preds_path
    preds_path.mkdir(parents=True, exist_ok=True)
    with open(preds_path / "predictions.jsonl", "w", encoding="utf-8") as pf:
        for i, (pr, tg) in enumerate(zip(predictions, targets)):
            pf.write(json.dumps({"idx": i, "pred": pr, "target": tg},
                                ensure_ascii=False) + "\n")

    metrics = compute_all_metrics(predictions, targets, dataset_name)
    metrics["checkpoint"] = str(checkpoint)
    metrics["trained"] = bool(getattr(inferencer, "has_checkpoint", False))

    out_dir = Path(output_dir)
    out_dir = out_dir if out_dir.is_absolute() else REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics.md").write_text(
        format_metrics_as_markdown(metrics), encoding="utf-8")

    print("\n" + format_metrics_as_markdown(metrics))
    if not metrics["trained"]:
        # 以前这里只打 warning 然后正常退出，调用方的 `&&` 判断会当成成功，
        # 于是一份随机初始化模型的指标被当作真实结果写进 evaluation_results。
        # 权重没加载上就是硬错误，必须让退出码非零。
        raise SystemExit(
            f"\n❌ 未加载有效权重（checkpoint={checkpoint}），以上指标无意义。\n"
            f"   HF LoRA 的 checkpoint 是目录（含 lora_adapter/ + projection.pt），"
            f"请指到具体那一层，例如 .../best_model.pt 而不是它的父目录。")
    print(f"\n✓ 结果已保存: {out_dir / 'metrics.json'}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Receipt-VLM 真实评估")
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "checkpoints" / "stage_b" / "best_model.pt"))
    parser.add_argument("--data", default=str(REPO_ROOT / "data" / "synthetic" / "train.jsonl"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--name", default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--output", default="evaluation_results/eval")
    parser.add_argument("--tokenizer", default=None,
                        help="显式指定 tokenizer（路线A=Qwen/Qwen2-1.5B；自制 mini 路线=tokenizers/receipt-bpe）")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.data, args.max_samples, args.name, args.device,
             args.max_new_tokens, args.temperature, args.output, args.tokenizer)


if __name__ == "__main__":
    main()
