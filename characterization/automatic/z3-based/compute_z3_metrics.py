import json, orjson, argparse, re
from typing import Any, Dict, List, Tuple, Optional, Set
from z3 import Solver, Int, Real, Bool, Not, Z3Exception

def jload(path:str):
    with open(path, "rb") as f:
        data = f.read()
    try:
        return orjson.loads(data)
    except Exception:
        rows = []
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(orjson.loads(line))
        return rows

def jdump(path:str, obj:Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _to_num(val: Any, prefer_real: bool):
    f = float(val)
    return f if prefer_real else (int(f) if f.is_integer() else f)

def compile_ir(
    ir: Dict[str, Any],
    skip_tracks: Optional[Set[str]] = None,
    negate_assert_tracks: Optional[Set[str]] = None,
    want_final_symbol: bool = True
) -> Tuple[Optional[Solver], List[str], Optional[Any], Dict[str,str]]:
    skip_tracks = skip_tracks or set()
    negate_assert_tracks = negate_assert_tracks or set()

    steps = (ir or {}).get("steps") or []
    variables = (ir or {}).get("variables") or []
    if not steps:
        return None, [], None, {}

    var_types = {v.get("name"): v.get("type","Int") for v in variables if v.get("name")}
    any_real = any(t == "Real" for t in var_types.values())

    versions: Dict[str,int] = {}
    def new_sym(name:str, ver:int):
        sort = Real if var_types.get(name,"Int") == "Real" else Int
        return sort(f"{name}@{ver}")

    def cur_sym(name:str):
        if name not in versions:
            versions[name] = 0
        return new_sym(name, versions[name])

    def bump(name:str):
        versions[name] = versions.get(name,0) + 1
        return new_sym(name, versions[name])

    def side_to_z3(side: Dict[str,Any], prefer_real: bool):
        if side is None: raise ValueError("missing side")
        if "var" in side and side["var"] is not None:
            return cur_sym(side["var"])
        if "const" in side and side["const"] is not None:
            return _to_num(side["const"], prefer_real)
        raise ValueError("side must have var/const")

    S = Solver()
    track_order: List[str] = []

    for name in var_types.keys():
        versions[name] = 0
        _ = new_sym(name, 0)

    def add_tracked(track: str, phi):
        S.assert_and_track(phi, Bool(track))
        track_order.append(track)

    for st in steps:
        try:
            track = st.get("track")
            op = st.get("op")
            if not track or not op or (track in skip_tracks):
                continue

            if op == "set":
                var = st.get("var")
                if var is None or "value" not in st: continue
                tgt = bump(var)
                rhs = _to_num(st["value"], var_types.get(var) == "Real")
                add_tracked(track, tgt == rhs)

            elif op == "update":
                var = st.get("var")
                ar = st.get("arith") or {}
                kind = ar.get("kind"); val = ar.get("value")
                if var is None or kind not in {"add","sub","mul","div"} or val is None: continue
                cur = cur_sym(var); tgt = bump(var)
                vv = _to_num(val, var_types.get(var) == "Real")
                if kind == "add":   phi = (tgt == cur + vv)
                elif kind == "sub": phi = (tgt == cur - vv)
                elif kind == "mul": phi = (tgt == cur * vv)
                else:               phi = (tgt == cur / vv)
                add_tracked(track, phi)

            elif op == "set_expr":
                var = st.get("var")
                if var is None and isinstance(st.get("lhs"), dict):
                    var = st["lhs"].get("var")
                expr = st.get("expr") or {}
                if var is None or "op" not in expr: continue
                tgt = bump(var)
                is_real = (var_types.get(var) == "Real")
                L = side_to_z3(expr.get("left") or {}, is_real or any_real)
                R = side_to_z3(expr.get("right") or {}, is_real or any_real)
                op2 = expr.get("op")
                if op2 == "add":   phi = (tgt == L + R)
                elif op2 == "sub": phi = (tgt == L - R)
                elif op2 == "mul": phi = (tgt == L * R)
                elif op2 == "div": phi = (tgt == L / R)
                else: continue
                add_tracked(track, phi)

            elif op in {"assert_eq","assert_ge","assert_le"}:
                lhs = st.get("lhs"); rhs = st.get("rhs")
                if lhs is None and ("var" in st or "value" in st):
                    v = st.get("var"); c = st.get("value")
                    lhs = {"var": v} if v is not None else None
                    rhs = {"const": c} if c is not None else None
                if lhs is None or rhs is None: continue
                is_real = False
                for sd in (lhs, rhs):
                    if sd and "const" in sd:
                        try:
                            x = float(sd["const"])
                            if not x.is_integer(): is_real = True
                        except Exception: pass
                L = side_to_z3(lhs, is_real or any_real)
                R = side_to_z3(rhs, is_real or any_real)
                if op == "assert_eq": phi = (L == R)
                elif op == "assert_ge": phi = (L >= R)
                else: phi = (L <= R)
                if track in negate_assert_tracks:
                    phi = Not(phi)
                add_tracked(track, phi)

        except Exception:
            continue

    fa_sym = cur_sym("final_answer") if want_final_symbol and "final_answer" in var_types else None
    return S, track_order, fa_sym, var_types

def compute_acs(S: Solver, tracks: List[str]) -> Optional[float]:
    if not S or not tracks: return None
    try:
        r = S.check()
    except Z3Exception:
        return None
    if str(r) == "unknown": return None
    if str(r) == "sat": return 1.0
    try:
        core = S.unsat_core()
        core_names = set(str(c) for c in core)
        k = sum(1 for t in tracks if t in core_names)
        return max(0.0, min(1.0, 1.0 - k / max(1, len(tracks))))
    except Exception:
        return 0.0

def list_assert_tracks(ir: Dict[str,Any]) -> List[str]:
    tracks = []
    for st in (ir or {}).get("steps", []):
        if st.get("op") in {"assert_eq","assert_ge","assert_le"} and st.get("track"):
            tracks.append(st["track"])
    return tracks

def compute_ear(ir: Dict[str,Any]) -> Optional[float]:
    a_tracks = list_assert_tracks(ir)
    if not a_tracks: return None
    forced = 0
    total = 0
    for t in a_tracks:
        S, tracks, _, _ = compile_ir(ir, skip_tracks={t}, negate_assert_tracks={t})
        if not S: continue
        try:
            r = S.check()
            total += 1
            if str(r) == "unsat":
                forced += 1
        except Z3Exception:
            continue
    if total == 0: return None
    return forced / total

def extract_predicted_answer(row:Dict[str,Any]) -> Optional[int]:
    for st in row.get("ir",{}).get("steps",[]):
        if st.get("op") == "assert_eq":
            lhs, rhs = st.get("lhs") or {}, st.get("rhs") or {}
            for a,b in ((lhs,rhs),(rhs,lhs)):
                if a.get("var") == "final_answer" and "const" in b and b["const"] is not None:
                    try:
                        f = float(b["const"])
                        if f.is_integer(): return int(f)
                    except Exception:
                        pass
    texts = []
    data = row.get("data") or {}
    for k in ("gcot","answer","pred","prediction","final","final_answer"):
        v = data.get(k)
        if isinstance(v,str): texts.append(v)
    for k in ("gcot","answer","pred","prediction","final","final_answer"):
        v = row.get(k)
        if isinstance(v,str): texts.append(v)
    txt = "\n".join(texts)
    m = re.findall(r"####\s*(-?\d+)", txt)
    if m: return int(m[-1])
    m = re.findall(r"(?:answer(?: is)?|final answer|total|thus|therefore|so)\D*(-?\d+)\b", txt, flags=re.I)
    if m: return int(m[-1])
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    for ln in reversed(lines[-3:]):
        m = re.findall(r"(-?\d+)\b", ln)
        if m: return int(m[-1])
    return None

from collections import defaultdict, deque
from typing import Set, Dict, Any, List

def find_bridge_vars(ir: Dict[str, Any]) -> Set[str]:
    steps: List[Dict[str, Any]] = (ir or {}).get("steps") or []
    G = defaultdict(set)
    for st in steps:
        if st.get("op") == "assert_eq":
            lhs, rhs = st.get("lhs") or {}, st.get("rhs") or {}
            lv, rv = lhs.get("var"), rhs.get("var")
            if lv and rv:
                G[lv].add(rv)
                G[rv].add(lv)
    for st in steps:
        if st.get("op") == "set_expr" and st.get("var") == "final_answer":
            expr = st.get("expr") or {}
            for side in ("left", "right"):
                s = expr.get(side) or {}
                v = s.get("var")
                if v:
                    G["final_answer"].add(v)
                    G[v].add("final_answer")
    bridges: Set[str] = set()
    if "final_answer" not in G:
        return {"final_answer"}
    q = deque(["final_answer"])
    bridges.add("final_answer")
    while q:
        u = q.popleft()
        for w in G[u]:
            if w not in bridges:
                bridges.add(w)
                q.append(w)
    return bridges

def detect_shortcuts(ir:Dict[str,Any], ans:Optional[int]) -> Set[str]:
    if ans is None: return set()
    bridges = find_bridge_vars(ir)
    short = set()
    for st in (ir or {}).get("steps",[]):
        if st.get("op") != "assert_eq": continue
        tr = st.get("track")
        lhs, rhs = st.get("lhs") or {}, st.get("rhs") or {}
        def is_c(x):
            try:
                return "const" in x and x["const"] is not None and float(x["const"]).is_integer() and int(float(x["const"])) == int(ans)
            except Exception:
                return False
        def is_bridge(x): return x.get("var") in bridges
        if (is_bridge(lhs) and is_c(rhs)) or (is_bridge(rhs) and is_c(lhs)):
            if tr: short.add(tr)
    return short

def compute_fas0(ir:Dict[str,Any], ans:Optional[int]) -> Optional[int]:
    if ans is None: return None
    shortcuts = detect_shortcuts(ir, ans)
    S, tracks, fa, _ = compile_ir(ir, skip_tracks=shortcuts)
    if not S or fa is None: return None
    try:
        S.add(fa != int(ans))
        r = S.check()
        return 1 if str(r) == "unsat" else 0
    except Z3Exception:
        return None

def compute_jss(ir:Dict[str,Any], ans:Optional[int]) -> Optional[float]:
    if ans is None: return None
    shortcuts = detect_shortcuts(ir, ans)
    S, tracks, fa, _ = compile_ir(ir, skip_tracks=shortcuts)
    if not S or not tracks or fa is None: return None
    try:
        S.add(fa != int(ans))
        r = S.check()
        if str(r) != "unsat":
            return None
        core = S.unsat_core()
        core_names = set(str(c) for c in core)
        k = sum(1 for t in tracks if t in core_names)
        return k / max(1, len(tracks))
    except Z3Exception:
        return None

def compute_rcr(ir:Dict[str,Any], ans:Optional[int]) -> Optional[float]:
    if ans is None: return None
    shortcuts = detect_shortcuts(ir, ans)
    S0, tracks0, _, _ = compile_ir(ir, skip_tracks=shortcuts)
    if not S0 or not tracks0: return None
    redund, total = 0, 0
    for t in tracks0:
        total += 1
        S, tracks, fa, _ = compile_ir(ir, skip_tracks=shortcuts | {t})
        if not S or fa is None: continue
        try:
            S.add(fa != int(ans))
            r = S.check()
            if str(r) == "unsat":
                redund += 1
        except Z3Exception:
            pass
    if total == 0: return None
    return redund / total

def process_rows(rows: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    out = []
    for row in rows:
        rowo = dict(row)
        ir = rowo.get("ir")
        print(f"Procesando: {ir}")
        if not isinstance(ir, dict):
            rowo.update({"acs":None,"ear":None,"fas0":None,"jss":None,"rcr":None,"dal":None})
            out.append(rowo); continue
        S, tracks, _, _ = compile_ir(ir)
        acs = compute_acs(S, tracks)
        ear = compute_ear(ir)
        ans = extract_predicted_answer(rowo)
        shortcuts = detect_shortcuts(ir, ans)
        dal = 1 if shortcuts else 0
        fas0 = compute_fas0(ir, ans)
        print("1")
        jss  = compute_jss(ir, ans)
        rcr  = compute_rcr(ir, ans)
        rowo.update({
            "acs": acs, "ear": ear,
            "fas0": fas0, "jss": jss, "rcr": rcr, "dal": dal
        })
        out.append(rowo)
        print(f"  ACS={acs} EAR={ear} FAS0={fas0} JSS={jss} RCR={rcr} DAL={dal}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = jload(args.input)
    if isinstance(rows, dict): rows = [rows]
    res = process_rows(rows)
    jdump(args.output, res)
    print(f"Escribí: {args.output} ({len(res)} filas)")

if __name__ == "__main__":
    main()

