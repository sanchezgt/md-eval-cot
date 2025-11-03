import argparse, json, re, sys
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional
import os
import torch
from collections import OrderedDict

def ensure_safetensors_dir(path_or_repo: str) -> str:
    p = Path(path_or_repo)
    if p.exists():
        if not any(p.glob("*.safetensors")):
            raise RuntimeError(
                f"El directorio '{p}' no contiene *.safetensors. "
                "Evita checkpoints .bin; usa safetensors."
            )
        return str(p)
    return path_or_repo

try:
    from bert_score import BERTScorer
except Exception as ex:
    print(f"[ERROR] bert-score no está instalado: {ex}", file=sys.stderr)
    raise

def normalize_math_md(t: str) -> str:
    t = str(t or "")
    t = re.sub(r"\$\$.*?\$\$|\$.*?\$", lambda m: " EQ(" + re.sub(r"\s+", " ", m.group(0).strip("$")) + ") ", t, flags=re.S)
    t = re.sub(r"\\\(|\\\)|\\\[|\\\]", " ", t)
    t = re.sub(r"```.*?```", " CODE ", t, flags=re.S)
    t = t.replace("\\times","*").replace("\\cdot","*").replace("\\div","/").replace("\\sqrt","sqrt")
    t = re.sub(r"\\frac{([^{}]+)}{([^{}]+)}", r"(\1/\2)", t)
    t = t.replace("≤","<=").replace("≥",">=").replace("–","-").replace("—","-")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def strip_gsm8k_final_answer(s: str) -> Tuple[str, bool]:
    s0 = str(s or "").strip()
    pat = re.compile(r"(.*?)\s*####\s*[-+]?[\d\.,]+(?:\s*)$", flags=re.S)
    m = pat.match(s0)
    if m:
        return m.group(1).strip(), True
    return s0, False

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cand_field", default="answer")
    ap.add_argument("--ref_field", default="gcot")
    ap.add_argument("--lang", default="en", choices=["en","es","fr","de","it","pt","zh","ja","ko"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bertscore_model", default="microsoft/deberta-xlarge-mnli")
    ap.add_argument("--bertscore_batch_size", type=int, default=8)
    ap.add_argument("--bertscore_idf", action="store_true")
    ap.add_argument("--bleurt_ckpt", default="BLEURT-20")
    ap.add_argument("--bleurt_batch_size", type=int, default=8)
    ap.add_argument("--strip_ref_final_answer", action="store_true")
    return ap.parse_args()

def resolve_device(arg_device: Optional[str]) -> str:
    if arg_device:
        return "cuda" if arg_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def prepare_bleurt(ckpt: str):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    try:
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

    try:
        import evaluate
    except Exception as ex:
        print(f"[ERROR] evaluate no está instalado: {ex}", file=sys.stderr)
        raise
    metric = evaluate.load("bleurt", checkpoint=ckpt)

    if prev_cuda is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda
    return metric

def collect_all_normalized_refs(in_path: Path, ref_field: str, strip_ref_final_answer: bool) -> List[str]:
    refs: List[str] = []
    with in_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                ex: Dict[str, Any] = json.loads(line)
            except Exception:
                continue
            ref_raw = str(ex.get(ref_field, "")).strip()
            if not ref_raw:
                continue
            if strip_ref_final_answer:
                ref_raw, _ = strip_gsm8k_final_answer(ref_raw)
            ref = normalize_math_md(ref_raw)
            if ref:
                refs.append(ref)
    return refs

def process_batch(batch_items: List[Tuple[str, str, Dict[str, Any]]],
                  batch_indices: List[int],
                  args, bleurt_metric, scorer: BERTScorer, device: str,
                  fout, n_out: int) -> int:
    if not batch_items:
        return n_out
    cands = [c for c, _, _ in batch_items]
    refs  = [r for _, r, _ in batch_items]
    P_list = R_list = F1_list = None
    try:
        P, R, F1 = scorer.score(cands, refs, batch_size=args.bertscore_batch_size, verbose=False)
        P_list, R_list, F1_list = P.tolist(), R.tolist(), F1.tolist()
    except Exception as ex_bert:
        print(f"[WARN] BERTScore falló: {ex_bert}", file=sys.stderr)
    bleurt_scores = None
    if bleurt_metric is not None:
        try:
            bleurt_scores = bleurt_metric.compute(predictions=cands, references=refs)["scores"]
        except Exception as ex_bl:
            print(f"[WARN] BLEURT falló: {ex_bl}", file=sys.stderr)
    for i, (_, _, ex) in enumerate(batch_items):
        if P_list is not None:
            ex["bertscore_precision"] = float(P_list[i])
            ex["bertscore_recall"]    = float(R_list[i])
            ex["bertscore_f1"]        = float(F1_list[i])
            ex["S_info_bert"]         = float(F1_list[i])
        else:
            ex["bertscore_precision"] = None
            ex["bertscore_recall"]    = None
            ex["bertscore_f1"]        = None
            ex["S_info_bert"]         = None
        if bleurt_scores is not None:
            s = float(bleurt_scores[i])
            ex["bleurt_score"]  = s
            ex["S_info_bleurt"] = s
        else:
            ex["bleurt_score"]  = None
            ex["S_info_bleurt"] = None
        ex_out = OrderedDict()
        ex_out["index"] = batch_indices[i]
        for k, v in ex.items():
            ex_out[k] = v
        fout.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
        n_out += 1
    batch_items.clear()
    batch_indices.clear()
    if torch.cuda.is_available() and (n_out % (args.bleurt_batch_size * 5) == 0):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return n_out

def main():
    args = parse_args()
    device = resolve_device(args.device)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args.bertscore_model = ensure_safetensors_dir(args.bertscore_model)
    try:
        scorer = BERTScorer(
            model_type=args.bertscore_model,
            lang=args.lang,
            rescale_with_baseline=True,
            device=device,
            batch_size=args.bertscore_batch_size,
        )
        scorer._model.eval()
        torch.set_grad_enabled(False)
        if device == "cuda":
            try:
                scorer._model.half()
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
            except Exception as ex:
                print(f"[WARN] No se pudo activar half precision: {ex}", file=sys.stderr)
    except Exception as ex:
        print(f"[ERROR] No se pudo cargar BERTScorer: {ex}", file=sys.stderr)
        raise
    in_path = Path(args.input)
    out_path = Path(args.output)
    if args.bertscore_idf:
        try:
            print("[INFO] Precomputando IDF global...", file=sys.stderr)
            all_refs = collect_all_normalized_refs(in_path, args.ref_field, args.strip_ref_final_answer)
            if not all_refs:
                print("[WARN] No hay referencias para IDF.", file=sys.stderr)
            else:
                scorer.compute_idf(all_refs)
                print(f"[INFO] IDF global listo. Nº refs: {len(all_refs)}", file=sys.stderr)
        except Exception as ex:
            print(f"[WARN] No se pudo precomputar IDF global: {ex}", file=sys.stderr)
    try:
        bleurt_metric = prepare_bleurt(args.bleurt_ckpt)
    except Exception as ex:
        print(f"[WARN] No se pudo cargar BLEURT ({args.bleurt_ckpt}): {ex}", file=sys.stderr)
        bleurt_metric = None
    n_in = n_out = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        batch_items: List[Tuple[str, str, Dict[str, Any]]] = []
        batch_indices: List[int] = []
        for idx, line in enumerate(fin, start=0):
            print('.', end='', flush=True)
            line = line.strip()
            if not line:
                continue
            n_in += 1
            ex: Dict[str, Any] = json.loads(line)
            cand_raw = str(ex.get(args.cand_field, "")).strip()
            ref_raw  = str(ex.get(args.ref_field,  "")).strip()
            ref_stripped = False
            if args.strip_ref_final_answer and ref_raw:
                ref_raw, ref_stripped = strip_gsm8k_final_answer(ref_raw)
            cand = normalize_math_md(cand_raw)
            ref  = normalize_math_md(ref_raw)
            ex["bertscore_model"] = args.bertscore_model
            ex["bleurt_ckpt"]     = args.bleurt_ckpt
            ex["info_details"] = {
                "normalized_gcot": (cand != cand_raw),
                "normalized_ref":  (ref != ref_raw),
                "ref_stripped_gsm8k": bool(ref_stripped),
                "lang": args.lang,
                "idf_global": bool(args.bertscore_idf)
            }
            if not cand or not ref:
                ex["bertscore_precision"] = None
                ex["bertscore_recall"]    = None
                ex["bertscore_f1"]        = None
                ex["S_info_bert"]         = None
                ex["bleurt_score"]        = None
                ex["S_info_bleurt"]       = None
                ex_out = OrderedDict()
                ex_out["index"] = idx
                for k, v in ex.items():
                    ex_out[k] = v
                fout.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
                n_out += 1
                continue
            batch_items.append((cand, ref, ex))
            batch_indices.append(idx)
            if len(batch_items) >= args.bleurt_batch_size:
                n_out = process_batch(batch_items, batch_indices, args, bleurt_metric, scorer, device, fout, n_out)
        n_out = process_batch(batch_items, batch_indices, args, bleurt_metric, scorer, device, fout, n_out)
    print(f"\n[DONE] Procesadas {n_out}/{n_in} filas. Salida: {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()

