import argparse, json, re, statistics, subprocess, sys, tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ap = argparse.ArgumentParser(description="goose-llama vs llama.cpp: same models, prompts, cores and priority")
ap.add_argument("--models", default=str(ROOT.parent / "models"))
ap.add_argument("--llama-bin", default=str(ROOT.parent / "llama.cpp" / "build" / "bin"))
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--mask", default="FFFF", help="process affinity: the 13900KF's 8 P-cores, both hyperthreads")
ap.add_argument("--strict-mask", default="0x5555", help="llama.cpp strict per-thread mask: one hyperthread per P-core")
ap.add_argument("--out", default=str(ROOT / "results" / "results.json"))
ap.add_argument("--skip-sweeps", action="store_true")
a = ap.parse_args()
Q = ("q8_0", "q4_k_m")
M = {q: str(Path(a.models) / f"qwen2.5-1.5b-instruct-{q}.gguf") for q in Q}
B = Path(a.llama_bin)
GL = str(ROOT / "llama.exe")
PIN = str(ROOT / "tools" / "pinhi.bat")
PP = (16, 32, 64, 128, 256, 512)
TH = (1, 2, 3, 4, 5, 6, 7, 8)
STRICT = ["-C", a.strict_mask, "--cpu-strict", "1", "--prio", "2"]

def run(argv, pin=False):
    if pin: argv = ["cmd", "/c", PIN, a.mask] + argv
    r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    return r.stdout + r.stderr

def runs(f):
    return [f() for _ in range(a.runs)]

def goose_gen(q, prompt, draft, n=256, t=None):
    out = run([GL, "-m", M[q], "-f", str(HERE / f"p_{prompt}.txt"), "-n", str(n), "--draft", str(draft)] + (["-t", str(t)] if t else []), pin=True)
    return float(re.search(r"gen \d+ tok ([\d.]+)", out).group(1))

def goose_pp(q, n):
    out = run([GL, "-m", M[q], "-f", str(HERE / f"p_pp{n}.txt"), "-n", "1", "--draft", "0"], pin=True)
    return float(re.search(rf"prompt {n} tok ([\d.]+) tok/s", out).group(1))

def goose_ppl(q):
    out = run([GL, "-m", M[q], "--ppl", str(HERE / "ppl.txt"), "-c", "128"], pin=True)
    return float(re.search(r"PPL = ([\d.]+) over \d+ tokens", out).group(1))

def parse_bench(out):
    rows, head = [], None
    for l in out.splitlines():
        if not l.startswith("|"): continue
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        if "t/s" in c: head = c; continue
        if not head or len(c) != len(head) or c[0].startswith("-"): continue
        r = dict(zip(head, c))
        m = re.match(r"(pp|tg)(\d+)$", r["test"])
        v = re.match(r"([\d.]+) ± ([\d.]+)", r["t/s"])
        if m and v: rows.append({"test": m.group(1), "n": int(m.group(2)), "t": int(r["threads"]), "mean": float(v.group(1)), "sd": float(v.group(2))})
    return rows

def llama_best(q, extra):
    best = {}
    for opts, pin in (([], False), ([], True), (STRICT, True)):
        for r in parse_bench(run([str(B / "llama-bench.exe"), "-m", M[q], "-ngl", "0", "-r", str(a.runs), "-nopo", "1"] + extra + opts, pin=pin)):
            k = (r["test"], r["n"], r["t"])
            if k not in best or r["mean"] > best[k]["mean"]: best[k] = r
    return best

def llama_lookup(q):
    out = run([str(B / "llama-lookup.exe"), "-m", M[q], "-f", str(HERE / "p_code.txt"), "-n", "256", "-ngl", "0", "-t", "8", "--temp", "0"] + STRICT, pin=True)
    return float(re.search(r"decoded\s+\d+ tokens in\s+[\d.]+ seconds, speed:\s+([\d.]+)", out).group(1))

def llama_ppl(q):
    out = run([str(B / "llama-perplexity.exe"), "-m", M[q], "-f", str(HERE / "ppl.txt"), "-c", "128", "-b", "128", "-ngl", "0", "-ctk", "f32", "-ctv", "f32", "--no-op-offload", "-t", "8"])
    return float(re.search(r"Final estimate: PPL = ([\d.]+)", out).group(1))

def weight_bytes(q):
    sys.path.insert(0, str(ROOT.parent / "llama.cpp" / "gguf-py"))
    from gguf import GGUFReader
    return int(sum(int(t.n_bytes) for t in GGUFReader(M[q]).tensors))

def bandwidth():
    best = {}
    for _ in range(5):
        for m in re.finditer(r"threads (\d+) gbs ([\d.]+)", run([str(ROOT / "tools" / "bw.exe")], pin=True)):
            t, g = int(m.group(1)), float(m.group(2))
            best[t] = max(best.get(t, 0), g)
    return {"t": sorted(best), "gbs": [best[t] for t in sorted(best)]}

def tok_parity():
    cases = ["Hello world", "I'm sure they'll say we've done it, DON'T you think?", "1234567 + 89 = 1234656",
             "  leading   spaces\n\n\ttabs\r\nand CRLF  ", "naïve café über Straße — “quotes” … 日本語のテキスト 中文 한국어",
             "emoji 🦆🔥👍🏽 mixed", "def f(x):\n    return x**2  # comment\n\n\n",
             "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", "a b　c", "!!! ??? ... ---", "x'S y'Re",
             "Ελληνικά, русский язык, العربية, עברית, हिन्दी", "price: $1,299.99 (20% off!) 2024-11-03T12:00Z",
             "{\"key\": [1, 2, {\"n\": null}]}\n", "    \n  \n x", "über-cool ÆØÅ æøå ﬁ ½ ²³"]
    d = Path(tempfile.mkdtemp())
    ok = 0
    for i, c in enumerate(cases):
        fp = d / f"{i}.txt"
        fp.write_bytes(c.encode())
        ref = run([str(B / "llama-tokenize.exe"), "-m", M["q8_0"], "-f", str(fp), "--ids", "--log-disable"]).strip().splitlines()
        ref = [int(x) for x in re.findall(r"-?\d+", ref[-1])] if ref else None
        got = run([GL, "-m", M["q8_0"], "-f", str(fp), "-n", "0", "-t", "1", "-v"])
        got = [int(x) for x in re.findall(r"-?\d+", next(l for l in got.splitlines() if l.startswith("tokens:")))]
        ok += ref == got
    return ok, len(cases)

res = {"machine": "Intel i9-13900KF (8P+16E), DDR5-6000 dual channel, Windows 11", "model": "Qwen2.5-1.5B-Instruct",
       "llama_cpp": "b10566 CPU (-ngl 0 -nopo 1 -t 8); best of unpinned, affinity-pinned, and strict one-thread-per-P-core at high priority",
       "goose_llama": "default workers (8 for Q8_0, 7 for Q4_K_M), pinned one per P-core, high priority",
       "runs": a.runs, "rows": [], "pp_sweep": {}, "t_sweep": {}, "bytes": {}}
for q in Q:
    print("==", q, flush=True)
    lb = llama_best(q, ["-t", "8", "-p", "128", "-n", "128"])
    gd = runs(lambda: goose_gen(q, "free", 0))
    gs = runs(lambda: goose_gen(q, "code", 7))
    ll = runs(lambda: llama_lookup(q))
    gp = runs(lambda: goose_pp(q, 128))
    tg, pp = lb[("tg", 128, 8)], lb[("pp", 128, 8)]
    res["rows"] += [
        {"model": q, "test": "decode", "goose": statistics.median(gd), "llama_cpp": tg["mean"], "goose_runs": gd, "llama_cpp_sd": tg["sd"]},
        {"model": q, "test": "speculative (code edit)", "goose": statistics.median(gs), "llama_cpp": statistics.median(ll), "goose_runs": gs, "llama_cpp_runs": ll},
        {"model": q, "test": "prefill", "goose": statistics.median(gp), "llama_cpp": pp["mean"], "goose_runs": gp, "llama_cpp_sd": pp["sd"]},
    ]
    res[f"ppl_{q}"] = {"goose": goose_ppl(q), "llama_cpp": llama_ppl(q)}
    res["bytes"][q] = weight_bytes(q)
    print(json.dumps(res["rows"][-3:]), flush=True)
    if a.skip_sweeps: continue
    lp = llama_best(q, ["-t", "8", "-p", ",".join(map(str, PP)), "-n", "0"])
    res["pp_sweep"][q] = {"n": list(PP), "goose": [runs(lambda: goose_pp(q, n)) for n in PP],
                          "llama_cpp": [[lp[("pp", n, 8)]["mean"], lp[("pp", n, 8)]["sd"]] for n in PP]}
    lt = llama_best(q, ["-t", ",".join(map(str, TH)), "-p", "0", "-n", "128"])
    res["t_sweep"][q] = {"t": list(TH), "goose": [runs(lambda: goose_gen(q, "free", 0, 128, t)) for t in TH],
                         "llama_cpp": [[lt[("tg", 128, t)]["mean"], lt[("tg", 128, t)]["sd"]] for t in TH]}
    print("sweeps done", q, flush=True)
if not a.skip_sweeps: res["bandwidth"] = bandwidth()
res["tokenizer"] = dict(zip(("match", "cases"), tok_parity()))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k in ("rows", "bandwidth", "bytes")}, indent=1))
