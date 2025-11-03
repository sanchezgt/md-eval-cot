import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict
from sentence_transformers import CrossEncoder
import numpy as np

def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

class CoherenceScorer:
    def __init__(self, model_name="enochlev/coherence-all-mpnet-base-v2", device=None, batch_size=8):
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size

    def score(self, question: str, gcot: str) -> Dict[str, float]:
        pairs = [[question, gcot]]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        mean_prob = float(np.mean(scores))
        std = float(np.std(scores))
        return {"mean_prob": mean_prob, "std": std, "n": len(scores)}

def summarize(results: List[Dict]) -> Dict:
    vals = [r["mean_prob"] for r in results if r.get("mean_prob") is not None]
    if not vals:
        return {}
    return {
        "n_total": len(results),
        "n_valid": len(vals),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }

def main():
    ap = argparse.ArgumentParser(description="Evalúa coherencia (pregunta vs gcot) en un dataset JSONL con CrossEncoder.")
    ap.add_argument("--input", type=str, help="Ruta al .jsonl con las COT")
    ap.add_argument("--output", type=str, default=None, help="Ruta de salida .jsonl con resultados por ítem")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size para inferencia")
    ap.add_argument("--device", type=str, default=None, help="cuda|mps|cpu (None = auto)")
    ap.add_argument("--max-items", type=int, default=None, help="Limitar a los primeros N items")
    args = ap.parse_args()

    dataset_path = Path(args.input)
    if not dataset_path.exists():
        print(f"ERROR: no existe el archivo {dataset_path}", file=sys.stderr)
        sys.exit(1)

    scorer = CoherenceScorer(
        model_name="enochlev/coherence-all-mpnet-base-v2",
        device=args.device,
        batch_size=args.batch_size
    )

    out_path = Path(args.output) if args.output else dataset_path.with_suffix(".coherence_results.jsonl")
    fout = open(out_path, "w", encoding="utf-8")

    results = []
    for idx, item in enumerate(load_dataset(str(dataset_path))):
        if args.max_items is not None and idx >= args.max_items:
            break
        if "question" not in item or "gcot" not in item:
            continue
        question, gcot = item["question"], item["gcot"]
        try:
            res = scorer.score(question, gcot)
            print(res)
        except Exception as e:
            res = {"error": str(e)}
        row = {"index": idx, "question": question, "gcot": gcot, **res}
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        results.append(row)
        if (idx + 1) % 20 == 0:
            print(f"[{idx+1}] mean_prob (último): {row.get('mean_prob')}  | acumulados: {idx+1}")

    fout.close()
    summary = summarize(results)
    print("\n==================")
    if not summary:
        print("No hay resultados válidos para resumir.")
        print(f"Archivo por-item guardado en: {out_path}")
        return
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print(f"\nArchivo con resultados por item: {out_path}")

if __name__ == "__main__":
    main()

