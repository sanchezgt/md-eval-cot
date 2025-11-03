import json
import sys
import argparse
from pathlib import Path
from math import sqrt
from typing import Dict, Any
from coherence_momentum_scorer import CoherenceMomentumScorer, CoherenceMomentumConfig

def load_dataset(file_name: str):
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def summarize(results):
    rows = []
    for r in results:
        try:
            mp = float(r["mean_prob"])
            sd = float(r["std"])
            n = int(round(float(r["n"])))
        except Exception:
            continue
        if n <= 1:
            continue
        rows.append({"mean_prob": mp, "std": sd, "n": n})
    if not rows:
        return {}

    N = sum(r["n"] for r in rows)
    wmean = sum(r["n"] * r["mean_prob"] for r in rows) / N

    num = sum((r["n"] - 1) * (r["std"] ** 2) for r in rows)
    num += sum(r["n"] * ((r["mean_prob"] - wmean) ** 2) for r in rows)
    denom = max(1, N - 1)
    pooled_var = num / denom
    pooled_std = sqrt(pooled_var)
    se = pooled_std / sqrt(N) if N > 0 else float("nan")
    H0 = 0.5
    z = (wmean - H0) / se if se > 0 else float("inf")
    ci_low = wmean - 1.96 * se
    ci_high = wmean + 1.96 * se

    return {
        "N_total_pairs": N,
        "mean_prob_weighted": wmean,
        "pooled_std": pooled_std,
        "se_mean": se,
        "z_vs_0.5": z,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "num_items": len(rows),
    }

def main():
    ap = argparse.ArgumentParser(description="Evalúa coherencia (coherence-momentum) en un dataset JSONL.")
    ap.add_argument("--input", type=str, help="Ruta al .jsonl con las COT")
    ap.add_argument("--field", type=str, default="gcot", help="Campo de texto (default: gcot)")
    ap.add_argument("--output", type=str, default=None, help="Ruta de salida .jsonl con resultados por item")
    ap.add_argument("--k", type=int, default=10, help="Número de degradaciones por texto (K)")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size para inferencia")
    ap.add_argument("--device", type=str, default=None, help="cuda|mps|cpu (None = auto)")
    ap.add_argument("--tau", type=float, default=1.0, help="Temperatura para sigmoide (si aplica)")
    ap.add_argument("--max-items", type=int, default=None, help="Limitar a los primeros N items")
    args = ap.parse_args()

    dataset_path = Path(args.input)
    if not dataset_path.exists():
        print(f"ERROR: no existe el archivo {dataset_path}", file=sys.stderr)
        sys.exit(1)

    cfg = CoherenceMomentumConfig(
        K=args.k,
        batch_size=args.batch_size,
        device=args.device,
    )
    scorer = CoherenceMomentumScorer(cfg)

    out_path = Path(args.output) if args.output else dataset_path.with_suffix(".coherence_results.jsonl")
    fout = open(out_path, "w", encoding="utf-8")

    results = []
    for idx, item in enumerate(load_dataset(str(dataset_path))):
        if args.max_items is not None and idx >= args.max_items:
            break
        if args.field not in item:
            continue
        text = item[args.field]
        try:
            res = scorer.score(text)
        except Exception as e:
            res = {"error": str(e)}
        row = {
            "index": idx,
            "mean_prob": res.get("mean_prob"),
            "std": res.get("std"),
            "z": res.get("z"),
            "n": res.get("n"),
        }
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        results.append(row)
        if (idx + 1) % 20 == 0:
            print(f"[{idx+1}] mean_prob (último): {row['mean_prob']:.3f} | acumulados: {idx+1}")

    fout.close()
    summary = summarize(results)
    print("\n==================")
    if not summary:
        print("No hay resultados válidos para resumir.")
        print(f"Archivo por-item guardado en: {out_path}")
        return
    for k, v in summary.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
    print(f"\nArchivo con resultados por item: {out_path}")

if __name__ == "__main__":
    main()

