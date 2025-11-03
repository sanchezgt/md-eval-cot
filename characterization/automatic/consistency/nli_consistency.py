import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from collections import OrderedDict

def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)

def harden_hf_env():
    os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
    os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", "https://huggingface.co")
    for k in ("HUGGINGFACE_CO_URL", "HF_API_ENDPOINT"):
        if os.environ.get(k, "").strip() in ("", "https://", "/"):
            os.environ.pop(k, None)
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(k):
            os.environ.pop(k, None)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    ap.add_argument("--lang", choices=["es", "en"], default="en")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--window_k", type=int, default=3)
    return ap.parse_args()

def normalize_math_md(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"\$\$.*?\$\$|\$.*?\$", " EQ ", t, flags=re.S)
    t = re.sub(r"```.*?```", " CODE ", t, flags=re.S)
    t = t.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    t = re.sub(r"\\frac{([^{}]+)}{([^{}]+)}", r"(\1/\2)", t)
    t = re.sub(r"[*_#`>]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def split_sentences_simple(t: str) -> List[str]:
    if not t:
        return []
    sents = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])", t.strip())
    sents = [s.strip() for s in sents if s.strip()]
    return sents or [t.strip()]

def build_nli(model_id: str, max_length: int, device_arg: str):
    use_cuda = torch.cuda.is_available()
    if not use_cuda or device_arg.lower() == "cpu":
        device = -1
    else:
        try:
            device = int(device_arg.split(":")[1]) if ":" in device_arg else 0
        except Exception:
            device = 0
    tok = AutoTokenizer.from_pretrained(model_id, model_max_length=max_length)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    nli = pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        top_k=None,
        function_to_apply="softmax",
        truncation=True,
        device=device,
    )
    try:
        id2label = {int(k): v for k, v in mdl.config.id2label.items()}
    except Exception:
        id2label = mdl.config.id2label
    return nli, id2label

def extract_probs(res: List[Dict[str, Any]], id2label: Dict[int, str] = None) -> Tuple[float, float, float]:
    p_ent, p_con, p_neu = 0.0, 0.0, 0.0
    def norm_label(lbl: str) -> str:
        s = (lbl or "").strip()
        up = s.upper()
        if up.startswith("LABEL_"):
            try:
                idx = int(up.split("_")[-1])
                if id2label and (idx in id2label or str(idx) in id2label):
                    s = id2label.get(idx, id2label.get(str(idx), s))
            except Exception:
                pass
        return str(s).lower()
    if isinstance(res, dict):
        res = [res]
    if isinstance(res, list) and res and isinstance(res[0], list):
        res = res[0]
    for d in res:
        lbl = norm_label(d.get("label", ""))
        sc = float(d.get("score", 0.0))
        if ("entail" in lbl) or ("support" in lbl):
            p_ent = sc
        elif ("contrad" in lbl) or ("refut" in lbl):
            p_con = sc
        elif ("neutral" in lbl) or ("unknown" in lbl) or ("not enough info" in lbl) or ("nei" in lbl):
            p_neu = sc
    s = p_ent + p_con + p_neu
    if s > 0:
        p_ent, p_con, p_neu = p_ent / s, p_con / s, p_neu / s
    return p_ent, p_con, p_neu

def consistency_q2g(nli, id2label, question: str, gcot_sents: List[str], batch_size: int) -> Tuple[float, float]:
    if not gcot_sents:
        return 1.0, 1.0
    batch = [{"text": question, "text_pair": s} for s in gcot_sents]
    results = nli(batch, batch_size=batch_size)
    p_con_list = []
    for res in results:
        _, p_con, _ = extract_probs(res, id2label=id2label)
        p_con_list.append(p_con)
    if not p_con_list:
        return 1.0, 1.0
    s_max = 1.0 - max(p_con_list)
    s_mean = 1.0 - (sum(p_con_list) / len(p_con_list))
    return float(s_max), float(s_mean)

def consistency_intra_local(nli, id2label, gcot_sents: List[str], k: int, batch_size: int) -> float:
    m = len(gcot_sents)
    if m <= 1:
        return 1.0
    vals = []
    for j in range(1, m):
        L = max(0, j - k)
        pairs = [{"text": gcot_sents[i], "text_pair": gcot_sents[j]} for i in range(L, j)]
        results = nli(pairs, batch_size=batch_size)
        pcon_max = 0.0
        for res in results:
            _, p_con, _ = extract_probs(res, id2label=id2label)
            if p_con > pcon_max:
                pcon_max = p_con
        vals.append(1.0 - pcon_max)
    return float(sum(vals) / len(vals)) if vals else 1.0

def main():
    args = parse_args()
    harden_hf_env()
    eprint(f"Cargando NLI: {args.model}")
    nli, id2label = build_nli(args.model, args.max_length, args.device)
    in_path = Path(args.input)
    out_path = Path(args.output)
    n_in, n_out = 0, 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin, start=0):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            ex = json.loads(line)
            q_raw = str(ex.get("question", "")).strip()
            g_raw = str(ex.get("gcot", "")).strip()
            q = normalize_math_md(q_raw)
            g = normalize_math_md(g_raw)
            sents_g = split_sentences_simple(g)
            S_q2g_max, S_q2g_mean = consistency_q2g(nli, id2label, q, sents_g, batch_size=args.batch_size)
            S_intra_local = consistency_intra_local(nli, id2label, sents_g, k=args.window_k, batch_size=args.batch_size)
            ex["nli_model"] = args.model
            ex["S_cons_q2g_max"] = S_q2g_max
            ex["S_cons_q2g_mean"] = S_q2g_mean
            ex["S_cons_intra_local"] = S_intra_local
            ex["cons_details"] = {
                "window_k": int(args.window_k),
                "n_sentences_gcot": len(sents_g),
                "normalized_question": (q != q_raw),
                "normalized_gcot": (g != g_raw),
            }
            ex_out = OrderedDict()
            ex_out["index"] = idx
            for k, v in ex.items():
                ex_out[k] = v
            fout.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
            n_out += 1
    eprint(f"Procesadas {n_out}/{n_in} filas. Salida: {out_path}")

if __name__ == "__main__":
    main()

