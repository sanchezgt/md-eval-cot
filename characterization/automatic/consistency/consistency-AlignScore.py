import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import OrderedDict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)

def normalize_math_md(t: str) -> str:
    t = re.sub(r"\$\$.*?\$\$|\$.*?\$", lambda m: " EQ(" + re.sub(r"\s+", " ", m.group(0).strip("$")) + ") ", t, flags=re.S)
    t = re.sub(r"\\\(|\\\)|\\\[|\\\]", " ", t)
    t = re.sub(r"```.*?```", " CODE ", t, flags=re.S)
    t = t.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/").replace("\\sqrt", "sqrt")
    t = re.sub(r"\\frac{([^{}]+)}{([^{}]+)}", r"(\1/\2)", t)
    t = t.replace("≤", "<=").replace("≥", ">=").replace("–", "-").replace("—", "-")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--backend", choices=["alignscore", "nli_fallback"], default="alignscore")
    ap.add_argument("--align_backbone", default="roberta-large")
    ap.add_argument("--align_ckpt", default="/opt/hf/alignscore/AlignScore-large.ckpt")
    ap.add_argument("--nli_model", default="facebook/bart-large-mnli")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()

def harden_hf_env():
    os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
    os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", "https://huggingface.co")
    for k in ("HUGGINGFACE_CO_URL", "HF_API_ENDPOINT"):
        if os.environ.get(k, "").strip() in ("", "https://", "/"):
            os.environ.pop(k, None)
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(k):
            os.environ.pop(k, None)

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or "").strip())
    if not text:
        return []
    parts = re.split(r"(?<=[\.\?\!\;\:])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 1]

def load_alignscore(backbone: str, ckpt_path: str, device_str: str, batch_size: int):
    from alignscore import AlignScore
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint de AlignScore no encontrado: {ckpt_path}")
    return AlignScore(model=backbone, batch_size=batch_size, device=device_str, ckpt_path=ckpt_path, evaluation_mode='nli_sp')

def load_nli(model_id: str, max_length: int, torch_device: int):
    tok = AutoTokenizer.from_pretrained(model_id, model_max_length=max_length)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id, use_safetensors=True)
    return pipeline("text-classification", model=mdl, tokenizer=tok, top_k=None, function_to_apply="softmax", truncation=True, device=torch_device)

def entail_contr_probs(nli_scores: List[Dict[str, Any]]) -> Tuple[float, float]:
    p_ent, p_con = 0.0, 0.0
    for d in nli_scores:
        lbl = d.get("label", "").lower()
        sc = float(d.get("score", 0.0))
        if "entail" in lbl or "support" in lbl:
            p_ent = sc
        elif "contrad" in lbl or "refut" in lbl:
            p_con = sc
    return p_ent, p_con

def main():
    args = parse_args()
    harden_hf_env()
    in_path = Path(args.input)
    out_path = Path(args.output)
    use_cuda = torch.cuda.is_available()
    device_str = args.device if use_cuda else "cpu"
    eprint(f"[INFO] Device: {device_str}")
    pipe_device = device_str if use_cuda else -1

    align_scorer = None
    backend_used = "nli_fallback"
    if args.backend == "alignscore":
        try:
            align_scorer = load_alignscore(args.align_backbone, args.align_ckpt, device_str, args.batch_size)
            backend_used = "alignscore"
            eprint(f"[INFO] Usando AlignScore ({args.align_backbone}) ckpt={args.align_ckpt}")
        except Exception as ex:
            eprint(f"[WARN] AlignScore no disponible ({ex}). Usando fallback NLI.")
            backend_used = "nli_fallback"

    nli_pipe = None
    if backend_used == "nli_fallback":
        eprint(f"[INFO] Cargando NLI fallback: {args.nli_model}")
        nli_pipe = load_nli(args.nli_model, args.max_length, pipe_device)

    n_in, n_out = 0, 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin, start=0):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            ex = json.loads(line)
            a = str(ex.get("question", "")).strip()
            b_raw = str(ex.get("gcot", "")).strip()
            r_raw = str(ex.get("response", "")).strip()
            b = normalize_math_md(b_raw)
            r = normalize_math_md(r_raw) if r_raw else ""

            if backend_used == "alignscore" and align_scorer is not None:
                s_q2g = align_scorer.score(contexts=[a], claims=[b])
                s_q2g = float(s_q2g[0] if isinstance(s_q2g, (list, tuple)) else s_q2g)
                s_g2r = None
                if r:
                    s_g2r_val = align_scorer.score(contexts=[b], claims=[r])
                    s_g2r = float(s_g2r_val[0] if isinstance(s_g2r_val, (list, tuple)) else s_g2r_val)
                ex["align_backend"] = "alignscore"
                ex["align_model_id"] = f"{args.align_backbone} @ {args.align_ckpt}"
                ex["S_support"] = s_q2g
                ex["S_imp"] = s_g2r
                ex["align_details"] = {"n_sentences_gcot": len(split_sentences(b)) if b else 0, "normalized_gcot": (b != b_raw), "normalized_response": (r != r_raw)}
            else:
                sents_b = split_sentences(b) or [b]
                batch_q2g = [{"text": a, "text_pair": s} for s in sents_b]
                outs_q2g = nli_pipe(batch_q2g)
                diffs_q2g = []
                for out in outs_q2g:
                    p_ent, p_con = entail_contr_probs(out)
                    diffs_q2g.append(p_ent - p_con)
                raw_q2g = (sum(diffs_q2g) / len(diffs_q2g)) if diffs_q2g else 0.0
                s_q2g = (raw_q2g + 1.0) / 2.0
                s_g2r = None
                raw_g2r = None
                if r:
                    sents_r = split_sentences(r) or [r]
                    batch_g2r = [{"text": b, "text_pair": t} for t in sents_r]
                    outs_g2r = nli_pipe(batch_g2r)
                    diffs_g2r = []
                    for out in outs_g2r:
                        p_ent, p_con = entail_contr_probs(out)
                        diffs_g2r.append(p_ent - p_con)
                    raw_g2r = (sum(diffs_g2r) / len(diffs_g2r)) if diffs_g2r else 0.0
                    s_g2r = (raw_g2r + 1.0) / 2.0
                ex["align_backend"] = "nli_fallback"
                ex["align_model_id"] = args.nli_model
                ex["S_support"] = float(s_q2g)
                ex["S_imp"] = (float(s_g2r) if s_g2r is not None else None)
                ex["align_details"] = {"n_sentences_gcot": len(sents_b), "n_sentences_resp": (len(split_sentences(r)) if r else 0), "raw_mean_diff_q2g": float(raw_q2g), "raw_mean_diff_g2r": (float(raw_g2r) if raw_g2r is not None else None), "normalized_gcot": (b != b_raw), "normalized_response": (r != r_raw)}

            ex_out = OrderedDict()
            ex_out["index"] = idx
            for k, v in ex.items():
                ex_out[k] = v
            fout.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
            n_out += 1

    eprint(f"[DONE] Procesadas {n_out}/{n_in} filas. Salida: {out_path}")

if __name__ == "__main__":
    main()

