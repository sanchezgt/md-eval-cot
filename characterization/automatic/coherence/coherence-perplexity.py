import argparse, json, sys, math
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def get_nested(d: Dict[str, Any], path: str) -> Optional[str]:
    cur = d
    for p in path.split('.'):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur

@torch.no_grad()
def perplexity(text: str,
               tokenizer: AutoTokenizer,
               model: AutoModelForCausalLM,
               device: torch.device,
               max_length: int = 1024,
               stride: int = 512) -> Dict[str, Any]:
    if text is None:
        return {"perplexity": None, "ppl_tokens": 0}
    text = text.strip()
    if not text:
        return {"perplexity": None, "ppl_tokens": 0}

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].squeeze(0).to(device)
    n_tokens = int(input_ids.numel())
    if n_tokens == 0:
        return {"perplexity": None, "ppl_tokens": 0}

    nll_total = 0.0
    count_tokens = 0

    for i in range(0, n_tokens, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, n_tokens)
        trg_len = end_loc - i
        if trg_len <= 0:
            break

        input_ids_slice = input_ids[begin_loc:end_loc]
        target_ids = input_ids_slice.clone()
        target_ids[:-trg_len] = -100

        outputs = model(input_ids_slice.unsqueeze(0), labels=target_ids.unsqueeze(0))
        nll_total += float(outputs.loss) * trg_len
        count_tokens += trg_len

        if end_loc == n_tokens:
            break

    ppl = math.exp(nll_total / max(1, count_tokens))
    return {"perplexity": ppl, "ppl_tokens": n_tokens}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--field", default="gcot")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        print(f"[ERROR] No existe {inp}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if (args.half and device.type == "cuda") else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code
    ).to(device)
    model.eval()

    total = sum(1 for _ in open(inp, "r", encoding="utf-8"))
    with open(inp, "r", encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(tqdm(fin, total=total, desc="Computing PPL"), start=0):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                fout.write(json.dumps({"index": idx, "raw": line, "perplexity": None, "ppl_model": args.model, "ppl_tokens": 0}, ensure_ascii=False) + "\n")
                continue

            text = get_nested(obj, args.field)
            res = perplexity(
                text=text,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=args.max_length,
                stride=args.stride
            )
            print(f"[DEBUG] Text len: {len(text) if text else 0}, PPL: {res['perplexity']}, Tokens: {res['ppl_tokens']}", file=sys.stderr)
            obj_out = {
                "index": idx,
                "perplexity": res["perplexity"],
                "ppl_model": args.model,
                "ppl_tokens": res["ppl_tokens"],
            }
            obj_out.update(obj)
            fout.write(json.dumps(obj_out, ensure_ascii=False) + "\n")

    print(f"[OK] Escrito: {out}")

if __name__ == "__main__":
    main()

