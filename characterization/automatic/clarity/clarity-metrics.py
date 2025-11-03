import argparse, json, math, re, unicodedata
from collections import Counter, defaultdict

def clean_text(text):
    text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)
    text = re.sub(r'~{1,2}(.*?)~{1,2}', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.M)
    text = re.sub(r'\$\$([^$]+)\$\$', r'\1', text)
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    text = re.sub(r'([a-zA-Z0-9]+)\^\{?([a-zA-Z0-9]+)\}?', r'\1 to the power of \2', text)
    return text.strip()

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def split_sentences(text):
    text = text.strip()
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z(])', text)
    sents = [s.strip() for s in parts if s and s.strip()]
    return sents or ([text] if text else [])

def tokenize_words(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)

VOWELS = "aeiouy"

def count_syllables_en(word):
    w = re.sub(r'[^A-Za-z]', '', word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    if w.endswith('e') and not (len(w) > 2 and w.endswith('le') and w[-3] not in VOWELS):
        w = w[:-1]
    count = 0
    prev_v = False
    for ch in w:
        v = (ch in VOWELS)
        if v and not prev_v:
            count += 1
        prev_v = v
    if len(w) > 2 and w.endswith('le') and w[-3] not in VOWELS:
        count += 1
    return max(1, count)

def readability_fre(text):
    sents = split_sentences(text)
    words = tokenize_words(text)
    syllables = sum(count_syllables_en(w) for w in words)
    ns, nw = len(sents), len(words)
    if ns == 0 or nw == 0:
        fre = 0.0
    else:
        fre = 206.835 - 1.015 * (nw / ns) - 84.6 * (syllables / nw)
    meta = {'sentences': ns, 'words': nw, 'syllables': syllables}
    if ns > 0 and nw > 0:
        meta.update({'w_per_s': nw / ns, 'syll_per_w': syllables / nw})
    return fre, meta

VAGUE_MULTI = {'of course', 'and so on', 'kind of', 'sort of'}
VAGUE_SINGLE = {
    'this','that','these','those','something','someone','somewhere','somehow','stuff','things','thing',
    'obviously','clearly','basically','evidently','surely','etc','it','they','them','itself','themselves',
    'approximately','about','around','roughly','soon','later','recently','nowadays','sometimes','often','frequently','rarely','usually','normally','typically'
}

def vagueness_density(text):
    toks = [t.lower() for t in tokenize_words(text)]
    i = 0
    count = 0
    while i < len(toks):
        if i + 2 < len(toks) and (toks[i] + ' ' + toks[i + 1] + ' ' + toks[i + 2]) in VAGUE_MULTI:
            count += 1
            i += 3
            continue
        if i + 1 < len(toks) and (toks[i] + ' ' + toks[i + 1]) in VAGUE_MULTI:
            count += 1
            i += 2
            continue
        if toks[i] in VAGUE_SINGLE:
            count += 1
            i += 1
            continue
        i += 1
    sents = split_sentences(text)
    rate = count / max(1, len(sents))
    return rate, {'vague_count': count, 'sentences': len(sents)}

def step_structure(text):
    strong = re.findall(r'(?im)^\s*(?:step|paso)\s*\d+\b', text)
    bullets = re.findall(r'(?im)^\s*(?:[-*•]\s+|\d+[\.\)]\s+)', text)
    inline = re.findall(r'(?i)\b(?:first|second|third|fourth|fifth|next|then|finally|lastly)\b', text)
    steps = len(strong) + len(bullets) + len(inline)
    sents = split_sentences(text)
    score = steps / max(1, len(sents))
    return score, {'steps': steps, 'strong': len(strong), 'bullets': len(bullets), 'inline': len(inline), 'sentences': len(sents)}

def extract_symbols(text):
    candidates = set()
    for m in re.findall(r'\$([^$]+)\$', text):
        for tok in re.findall(r'[A-Za-z][A-Za-z0-9_]*', m):
            candidates.add(tok)
    for tok in re.findall(r'\b[A-Za-z]_\w+\b|\b[A-Za-z]\b', text):
        candidates.add(tok)
    for tok in re.findall(r'\b[A-Z]{2,}\b', text):
        candidates.add(tok)
    return candidates

def is_defined(symbol, text):
    patts = [
        rf'(?i)\blet\s+{re.escape(symbol)}\b',
        rf'(?i)\bdenote\s+{re.escape(symbol)}\b',
        rf'(?i)\bdefine\s+{re.escape(symbol)}\b',
        rf'(?i)\bwhere\s+{re.escape(symbol)}\b',
        rf'\b{re.escape(symbol)}\s*(?::=|=)'
    ]
    return any(re.search(p, text) for p in patts)

def definition_coverage(text):
    syms = extract_symbols(text)
    if not syms:
        return 1.0, {'symbols_used': 0, 'symbols_defined': 0, 'symbols': [], 'defined': []}
    defined = [s for s in syms if is_defined(s, text)]
    score = len(defined) / len(syms)
    return score, {'symbols_used': len(syms), 'symbols_defined': len(defined), 'symbols': sorted(syms), 'defined': sorted(defined)}

UNITS = {
    '%','percent','percentage','usd','eur','gbp','cop','$','€','£','¥',
    'meter','meters','metre','metres','m','km','cm','mm','nm',
    'inch','inches','in','ft','feet','foot','mi','mile','miles',
    's','sec','secs','second','seconds','ms','millisecond','milliseconds','us','microsecond','ns','nanosecond',
    'min','mins','minute','minutes','h','hr','hrs','hour','hours','day','days','week','weeks','month','months','year','years',
    'kg','g','mg','lb','lbs','ounce','oz','ton','tons',
    'c','f','°c','°f','k','w','kw','mw','gw','wh','kwh','mwh','gwh',
    'v','mv','kv','a','ma','b','kb','mb','gb','tb','pb','token','tokens'
}
CURRENCIES = {'$','€','£','¥'}
CURRENCY_WORDS = {'usd','eur','gbp','cop','cad','aud','mxn','inr','jpy','cny'}

def units_explicitness(text):
    tokens = re.findall(r'\b\w+%?|\$|\€|\£|\¥|°C|°F', text)
    numbers = []
    for i, t in enumerate(tokens):
        if re.fullmatch(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:st|nd|rd|th)?%?', t):
            numbers.append((i, t))
    needs = 0
    have = 0
    for i, t in numbers:
        raw = t.lower()
        if re.search(r'(st|nd|rd|th)$', raw):
            continue
        if raw.endswith('%') or (i + 1 < len(tokens) and tokens[i + 1].lower() == 'percent'):
            have += 1
            continue
        if i > 0 and tokens[i - 1] in CURRENCIES:
            have += 1
            continue
        if i + 1 < len(tokens) and tokens[i + 1].lower() in CURRENCY_WORDS:
            have += 1
            continue
        try:
            val = int(re.sub(r'[^\d]', '', raw))
            if 1900 <= val <= 2099:
                continue
        except:
            pass
        if i > 0 and tokens[i - 1].lower() in ('step', 'paso'):
            continue
        needs += 1
        nxt = tokens[i + 1].lower() if i + 1 < len(tokens) else ''
        nxt2 = tokens[i + 2].lower() if i + 2 < len(tokens) else ''
        prev = tokens[i - 1].lower() if i - 1 >= 0 else ''
        if nxt in UNITS or nxt2 in UNITS or prev in UNITS:
            have += 1
    score = 1.0 if needs == 0 else have / needs
    return score, {'numbers_total': len(numbers), 'numbers_need_unit': needs, 'numbers_with_unit': have}

SUBORDINATORS = {'because','although','that','if','while','since','unless','whereas','though','when','whenever','after','before','once','until','as'}

def syntactic_simplicity_lite(text):
    sents = split_sentences(text)
    words = tokenize_words(text)
    ns = max(1, len(sents))
    nw = max(1, len(words))
    avg_len = nw / ns
    comma_rate = len(re.findall(r'[;,]', text)) / nw
    subs = sum(1 for w in words if w.lower() in SUBORDINATORS) / ns
    complexity = 0.5 * avg_len + 0.3 * comma_rate * 100 + 0.2 * subs
    return complexity, {'avg_sent_len': avg_len, 'comma_per_word': comma_rate, 'subord_per_sent': subs}

STOPWORDS = set("""
the a an of to in for on at by with from as about into through during including until against among throughout despite towards upon concerning regarding via over before after between without within along following across behind beyond plus except but up out around down off above near more most less least many much such other same any each another both either neither one two three first second third next then finally lastly prior subsequent
""".split())

def terminology_consistency_lite(text, top_k=8, min_freq=2):
    words = [w.lower() for w in tokenize_words(text)]
    counts = Counter()
    for n in range(2, 5):
        for i in range(len(words) - n + 1):
            gram = tuple(words[i:i + n])
            if gram[0] in STOPWORDS or gram[-1] in STOPWORDS:
                continue
            counts[gram] += 1
    common = [g for g, c in counts.most_common(50) if c >= min_freq][:top_k]
    groups = defaultdict(Counter)
    text_low = text.lower()
    for gram in common:
        surface = ' '.join(gram)
        canonical = re.sub(r'[\s\-]', '', surface)
        variants = {
            surface,
            surface.replace(' ', '-'),
            surface.replace(' ', ''),
            surface.replace(' ', '‐'),
        }
        for pat in variants:
            groups[canonical][pat] += len(re.findall(re.escape(pat), text_low))
    entropies = []
    for _, cnts in groups.items():
        total = sum(cnts.values())
        forms = [f for f, c in cnts.items() if c > 0]
        k = len(forms)
        if total == 0 or k <= 1:
            entropies.append(0.0)
        else:
            H = -sum((c / total) * math.log((c / total) + 1e-12) for c in cnts.values() if c > 0)
            entropies.append(H / math.log(k))
    tc = 1.0 - (sum(entropies) / len(entropies)) if entropies else 1.0
    return tc, {'ngrams_considered': [' '.join(g) for g in common], 'groups': {k: dict(v) for k, v in groups.items()}}

def percentile_bounds(values, p_low=0.05, p_high=0.95):
    if not values:
        return (0.0, 1.0)
    vs = sorted(values)
    def pct(p):
        k = (len(vs) - 1) * p
        f = int(k)
        c = min(len(vs) - 1, f + 1)
        if f == c:
            return vs[f]
        return vs[f] * (c - k) + vs[c] * (k - f)
    return pct(p_low), pct(p_high)

def compute_all_raw(text):
    fre, m1 = readability_fre(text)
    vd_raw, m2 = vagueness_density(text)
    se_raw, m3 = step_structure(text)
    dc_raw, m4 = definition_coverage(text)
    ue_raw, m5 = units_explicitness(text)
    ss_raw, m6 = syntactic_simplicity_lite(text)
    tc_raw, m7 = terminology_consistency_lite(text)
    raw = {'fre': fre,'vd_rate': vd_raw,'se_ratio': se_raw,'dc_ratio': dc_raw,'ue_ratio': ue_raw,'ss_complexity': ss_raw,'tc_consistency': tc_raw}
    meta = {'readability': m1,'vagueness': m2,'steps': m3,'definitions': m4,'units': m5,'syntax': m6,'terminology': m7}
    return raw, meta

DEFAULT_WEIGHTS = {'rd':0.18,'vd':0.12,'se':0.18,'dc':0.16,'ue':0.12,'ss_lite':0.12,'tc_lite':0.12}

def normalize_and_score(all_raw):
    fre_vals = [r['fre'] for r in all_raw]
    vd_vals = [r['vd_rate'] for r in all_raw]
    ss_vals = [r['ss_complexity'] for r in all_raw]
    tc_vals = [r['tc_consistency'] for r in all_raw]
    p5_fre, p95_fre = percentile_bounds(fre_vals)
    p5_vd, p95_vd = percentile_bounds(vd_vals)
    p5_ss, p95_ss = percentile_bounds(ss_vals)
    p5_tc, p95_tc = percentile_bounds(tc_vals)
    scores = []
    for r in all_raw:
        if p95_fre > p5_fre:
            rd = (r['fre'] - p5_fre) / (p95_fre - p5_fre)
        else:
            rd = (r['fre'] + 30.0) / 130.0
        rd = max(0.0, min(1.0, rd))
        if p95_vd > p5_vd:
            vd = 1.0 - max(0.0, min(1.0, (r['vd_rate'] - p5_vd) / (p95_vd - p5_vd)))
        else:
            vd = 1.0 - min(1.0, r['vd_rate'])
        if p95_ss > p5_ss:
            ss = 1.0 - max(0.0, min(1.0, (r['ss_complexity'] - p95_ss) / (p95_ss - p5_ss)))
        else:
            ss = 1.0 - min(1.0, r['ss_complexity'] / 50.0)
        se = max(0.0, min(1.0, r['se_ratio']))
        dc = max(0.0, min(1.0, r['dc_ratio']))
        ue = max(0.0, min(1.0, r['ue_ratio']))
        if p95_tc > p5_tc:
            tc = max(0.0, min(1.0, (r['tc_consistency'] - p5_tc) / (p95_tc - p5_tc)))
        else:
            tc = max(0.0, min(1.0, r['tc_consistency']))
        scores.append({'rd': rd,'vd': vd,'se': se,'dc': dc,'ue': ue,'ss_lite': ss,'tc_lite': tc})
    return scores

def aggregate_clx(score, weights=None):
    w = weights or DEFAULT_WEIGHTS
    return sum(w.get(k, 0.0) * score.get(k, 0.0) for k in w)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Input JSONL file.')
    ap.add_argument('--output', required=True, help='Output JSONL file.')
    ap.add_argument('--field', default='gcot', help='Field to evaluate (default: gcot).')
    ap.add_argument("--device", type=str, default=None, help="cuda|mps|cpu (None = auto)")
    args = ap.parse_args()
    rows = []
    all_raw = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get

