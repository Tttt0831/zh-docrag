#!/usr/bin/env python
"""视觉式一侧：ColQwen2.5 把页面图编码成多向量，查询编码成向量，MaxSim 打分。

与解析式共用 scripts/metrics.py 的指标函数，保证两边算的是同一个数。
"""
import argparse, glob, json, time
from pathlib import Path
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default=str(REPO / "data/pages/*.png"))
    ap.add_argument("--queries", default=str(REPO / "data/queries_mvp.jsonl"))
    ap.add_argument("--out", default=str(REPO / "data/visual_scores.json"))
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    # 必须用本地快照路径，不能用 repo id。
    # transformers 4.57.6 按 repo id 解析 additional_chat_templates/ 下的模板文件时
    # 会返回 None，随后在 processing_utils.py 里 open(None) 崩掉：
    #   TypeError: expected str, bytes or os.PathLike object, not NoneType
    # 文件本身在本地快照里是齐的（chat_template.jinja + additional_chat_templates/
    # sentence_transformers.jinja），给本地路径即可绕过那套解析。
    import glob as _g
    snaps = sorted(_g.glob(str(Path.home() / ".cache/huggingface/hub/models--vidore--colqwen2.5-v0.2/snapshots/*/")))
    if not snaps:
        raise SystemExit("未找到 colqwen2.5-v0.2 本地快照，先 hf download")
    name = snaps[0]
    t0 = time.time()
    model = ColQwen2_5.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    proc = ColQwen2_5_Processor.from_pretrained(name)
    print(f"  模型加载 {time.time()-t0:.1f}s", flush=True)

    paths = sorted(glob.glob(args.pages))
    print(f"  页面 {len(paths)} 张", flush=True)

    embs = []
    t1 = time.time()
    for i in range(0, len(paths), args.batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + args.batch]]
        batch = proc.process_images(imgs).to(model.device)
        with torch.no_grad():
            e = model(**batch)
        embs.extend(list(torch.unbind(e.to("cpu"))))
        print(f"   编码 {min(i+args.batch,len(paths))}/{len(paths)}", flush=True)
    dt = time.time() - t1
    print(f"  页面编码耗时 {dt:.1f}s（{dt/len(paths):.2f}s/页）", flush=True)
    print(f"  单页向量形状 {tuple(embs[0].shape)}  即 {embs[0].shape[0]} 个 patch × {embs[0].shape[1]} 维", flush=True)

    qs = [json.loads(l) for l in open(args.queries)]
    qb = proc.process_queries([q["query"] for q in qs]).to(model.device)
    with torch.no_grad():
        qe = model(**qb).to("cpu")

    scores = proc.score_multi_vector(qe, embs)   # MaxSim
    out = {"pages": [Path(p).stem for p in paths],
           "qids": [q["qid"] for q in qs],
           "scores": scores.tolist()}
    json.dump(out, open(args.out, "w"))
    print(f"  打分矩阵 {tuple(scores.shape)} 已存 {args.out}", flush=True)


if __name__ == "__main__":
    main()
