#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, re
from pathlib import Path
from typing import Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from collections import OrderedDict

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
    ap.add_argument("--model", default="manueldeprada/FactCC")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()

FACTUAL = {"factual","correct","entail","entailment","supported","true","yes"}
NONFACT = {"incorrect","non-factual","nonfactual","contradiction","refuted","unsupported","false","no"}

def norm(lbl: str) -> str:
    return re.sub(r"[\s\-]+", "", str(lbl).strip().lower())

def pick_probs(res):
    if isinstance(res, dict):
        res = [res]
    p_fact = p_non = 0.0
    labels = [d["label"] for d in res]
    for d in res:
        lbl = norm(d["label"])
        sc = float(d["score"])
        if lbl in FACTUAL:
            p_fact = max(p_fact, sc)
        elif lbl in NONFACT:
            p_non = max(p_non, sc)
    if p_fact == 0.0 and p_non == 0.0 and len(res) >= 2:
        res_sorted = sorted(res, key=lambda x: x["score"], reverse=True)
        p_fact = float(res_sorted[0]["score"])
        p_non = float(res_sorted[1]["score"])
    s = p_fact + p_non
    if s > 0:
        p_fact, p_non = p_fact / s, p_non / s
    return p_fact, p_non, labels

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else -1
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForSequenceClassification.from_pretrained(args.model, use_safetensors=True)
    clf = pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        top_k=None,
        function_to_apply="softmax",
        truncation=True,
        device=device,
    )
    in_path = Path(args.input)
    out_path = Path(args.output)
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin, start=0):
            if not line.strip():
                continue
            ex = json.loads(line)
            src = str(ex.get("gcot", "")).strip()
            summ = str(ex.get("response", "")).strip()
            summ = normalize_math_md(summ)
            res = clf({"text": src, "text_pair": summ})
            p_fact, p_non, labels = pick_probs(res)
            ex["factcc_model"] = args.model
            ex["factcc_labels"] = labels
            ex["factcc_p_factual"] = p_fact
            ex["factcc_p_nonfactual"] = p_non
            ex["S_fact"] = p_fact
            ex_out = OrderedDict()
            ex_out["index"] = idx
            for k, v in ex.items():
                ex_out[k] = v
            fout.write(json.dumps(ex_out, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

