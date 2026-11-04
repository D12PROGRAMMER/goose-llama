import argparse, json, re, statistics, subprocess, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ap = argparse.ArgumentParser(description="goose-llama --gpu vs llama.cpp CUDA (-ngl 99): same models, prompts and GPU")
ap.add_argument("--models", default=str(ROOT.parent / "models"))
ap.add_argument("--llama-bin", default=str(ROOT.parent / "llama.cpp" / "build" / "bin"))
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--out", default=str(ROOT / "results" / "results_gpu.json"))
a = ap.parse_args()
Q = ("q8_0", "q4_k_m")
M = {q: str(Path(a.models) / f"qwen2.5-1.5b-instruct-{q}.gguf") for q in Q}
B = Path(a.llama_bin)
GL = str(ROOT / "llama.exe")
PP = (128, 512, 1024, 2048)

def run(argv):
    r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    return r.stdout + r.stderr

def med(f):
    v = [f() for _ in range(a.runs)]
    return statistics.median(v), v

def goose(q, prompt, n, draft, key, ctx=4096):
    out = run([GL, "-m", M[q], "-f", str(HERE / f"p_{prompt}.txt"), "-n", str(n), "--gpu", "--draft", str(draft), "-c", str(ctx)])
    return float(re.search(key, out).group(1))

def bench(q, extra):
    out = run([str(B / "llama-bench.exe"), "-m", M[q], "-ngl", "99", "-r", str(a.runs), *extra])
    r = {}
    for l in out.splitlines():
        m = re.search(r"\|\s+((?:pp|tg)\d+(?: @ d\d+)?)\s+\|\s+([\d.]+) ± ([\d.]+)", l)
        if m: r[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return r

def lookup(q):
    out = run([str(B / "llama-lookup.exe"), "-m", M[q], "-f", str(HERE / "p_code.txt"), "-n", "256", "-ngl", "99", "--temp", "0"])
    return float(re.search(r"decoded\s+\d+ tokens in\s+[\d.]+ seconds, speed:\s+([\d.]+)", out).group(1))

res = {"gpu": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]).strip(),
       "model": "Qwen2.5-1.5B-Instruct", "runs": a.runs,
       "llama_cpp": "CUDA build, -ngl 99; llama-bench pp128..pp2048, tg128 and tg128 @ d2048, llama-lookup on test/p_code.txt",
       "goose_llama": "--gpu: decode test/p_free.txt 256 tokens --draft 0 and 128 after test/p_pp2048.txt, prefill test/p_ppN.txt, "
                      "speculative test/p_code.txt --draft 7",
       "rows": []}
for q in Q:
    print("==", q, flush=True)
    lb = bench(q, ["-p", ",".join(map(str, PP)), "-n", "128"]) | bench(q, ["-p", "0", "-n", "128", "-d", "2048"])
    rows = []
    gd, gdr = med(lambda: goose(q, "free", 256, 0, r"gen \d+ tok ([\d.]+)"))
    rows.append({"model": q, "test": "decode", "goose": gd, "goose_runs": gdr, "llama_cpp": lb["tg128"][0], "llama_cpp_sd": lb["tg128"][1]})
    gd, gdr = med(lambda: goose(q, "pp2048", 128, 0, r"gen \d+ tok ([\d.]+)"))
    rows.append({"model": q, "test": "decode @ 2048", "goose": gd, "goose_runs": gdr, "llama_cpp": lb["tg128 @ d2048"][0], "llama_cpp_sd": lb["tg128 @ d2048"][1]})
    for n in PP:
        gp, gpr = med(lambda: goose(q, f"pp{n}", 1, 0, r"prompt \d+ tok ([\d.]+)"))
        rows.append({"model": q, "test": f"prefill {n}", "goose": gp, "goose_runs": gpr, "llama_cpp": lb[f"pp{n}"][0], "llama_cpp_sd": lb[f"pp{n}"][1]})
    gs, gsr = med(lambda: goose(q, "code", 256, 7, r"gen \d+ tok ([\d.]+)"))
    ll, llr = med(lambda: lookup(q))
    rows.append({"model": q, "test": "speculative (code edit)", "goose": gs, "goose_runs": gsr, "llama_cpp": ll, "llama_cpp_runs": llr})
    res["rows"] += rows
    for r in rows: print(f"{r['test']:>24}: goose {r['goose']:.1f}  llama.cpp {r['llama_cpp']:.1f}  {r['goose'] / r['llama_cpp']:.2f}x", flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))
