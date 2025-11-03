import argparse
import json
import sys
from pathlib import Path
import numpy as np

from coherence_embedding import EmbeddingCoherence


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def summarize(results):
    vals = [r["mean_similarity"] for r in results if r.get("mean_similarity") is not None]
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
    ap = argparse.ArgumentParser(description="Evalúa coherencia local (embedding-based) en un dataset JSONL.")
    ap.add_argument("--input", type=str, help="Ruta al .jsonl con las COT")
    ap.add_argument("--field", type=str, default="gcot", help="Campo de texto a evaluar (default: gcot)")
    ap.add_argument("--output", type=str, default=None, help="Ruta de salida .jsonl con resultados por ítem")
    ap.add_argument("--device", type=str, default=None, help="cuda|mps|cpu (None = auto)")
    ap.add_argument("--max-items", type=int, default=None, help="Limitar a los primeros N items")
    args = ap.parse_args()

    dataset_path = Path(args.input)
    if not dataset_path.exists():
        print(f"ERROR: no existe el archivo {dataset_path}", file=sys.stderr)
        sys.exit(1)

    scorer = EmbeddingCoherence(device=args.device)

    out_path = Path(args.output) if args.output else dataset_path.with_suffix(".embeddings_results.jsonl")
    fout = open(out_path, "w", encoding="utf-8")

    results = []
    for idx, item in enumerate(load_dataset(str(dataset_path))):
        if args.max_items is not None and idx >= args.max_items:
            break
        if args.field not in item:
            continue

        gcot = item[args.field]
        try:
            res = scorer.score(gcot)
        except Exception as e:
            res = {"error": str(e)}

        row = {
            "index": idx,
            "gcot": gcot,
            **res
        }
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        results.append(row)

        if (idx + 1) % 20 == 0:
            print(f"[{idx+1}] mean_similarity (último): {row.get('mean_similarity')}  | acumulados: {idx+1}")

    fout.close()

    # resumen global
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
