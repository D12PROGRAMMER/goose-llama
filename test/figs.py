import json, statistics, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

ROOT = Path(__file__).resolve().parent.parent
ARGS = [x for x in sys.argv[1:] if not x.startswith("--")]
R = json.loads(Path(ARGS[0] if ARGS else ROOT / "results" / "results.json").read_text())
OUT = Path(ARGS[1]) if len(ARGS) > 1 else ROOT / "results"
NAMES = {"q8_0": "Q8_0", "q4_k_m": "Q4_K_M"}
G = dict(color="#1f5fa8", label="goose-llama", marker="o", ms=4.5, mfc="#1f5fa8", mec="#1f5fa8")
L = dict(color="#c4511c", label="llama.cpp", marker="s", ms=4.5, mfc="white", mec="#c4511c")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 2, "ytick.minor.size": 2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.1, "errorbar.capsize": 2,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "svg.fonttype": "path",
    "figure.dpi": 100, "savefig.dpi": 300,
})

def gstat(v):
    m = statistics.median(v)
    return m, m - min(v), max(v) - m

def panel(ax, tag):
    ax.text(-0.16, 1.02, f"({tag})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")

def save(fig, name):
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)

def row(q, t):
    return next(r for r in R["rows"] if r["model"] == q and r["test"] == t)

def fig1():
    tests = (("decode", "Decode, 256 tokens"), ("speculative (code edit)", "Speculative decode, code edit"), ("prefill", "Prefill, 128-token prompt"))
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.45))
    w = 0.36
    for k, ((t, title), ax) in enumerate(zip(tests, axs)):
        top = 0
        for i, q in enumerate(("q8_0", "q4_k_m")):
            r = row(q, t)
            gm, gl, gh = gstat(r["goose_runs"])
            if "llama_cpp_runs" in r: lm, ll, lh = gstat(r["llama_cpp_runs"])
            else: lm, ll, lh = r["llama_cpp"], r.get("llama_cpp_sd", 0), r.get("llama_cpp_sd", 0)
            ax.bar(i - w / 2, gm, w, color=G["color"], edgecolor="black", linewidth=0.5, yerr=[[gl], [gh]], error_kw=dict(elinewidth=0.7, ecolor="black"), label=G["label"] if i == 0 else None)
            ax.bar(i + w / 2, lm, w, color="white", edgecolor=L["color"], hatch="////", linewidth=0.8, yerr=[[ll], [lh]], error_kw=dict(elinewidth=0.7, ecolor="black"), label=L["label"] if i == 0 else None)
            hi = max(gm + gh, lm + lh)
            ax.text(i, hi * 1.04, f"{gm / lm:.2f}$\\times$", ha="center", va="bottom", fontsize=8)
            top = max(top, hi)
        ax.set_xticks([0, 1], [NAMES["q8_0"], NAMES["q4_k_m"]])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, top * 1.22)
        ax.set_title(title, pad=4)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", which="both", top=False)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(NullLocator())
        if k == 0: ax.set_ylabel("Throughput (tokens s$^{-1}$)")
        panel(ax, "abc"[k])
    axs[0].legend(loc="upper left", handlelength=1.4, borderaxespad=0.2)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig1_throughput")

def fig2():
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.6))
    for k, (q, ax) in enumerate(zip(("q8_0", "q4_k_m"), axs)):
        d = R["pp_sweep"][q]
        n = d["n"]
        gs = [gstat(v) for v in d["goose"]]
        ax.errorbar(n, [g[0] for g in gs], yerr=[[g[1] for g in gs], [g[2] for g in gs]], **G)
        ax.errorbar(n, [v[0] for v in d["llama_cpp"]], yerr=[v[1] for v in d["llama_cpp"]], linestyle="--", **L)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(n))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([str(x) for x in n])
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.set_xlabel("Prompt length (tokens)")
        if k == 0: ax.set_ylabel("Prefill throughput (tokens s$^{-1}$)")
        ax.set_title(NAMES[q], pad=4)
        ax.set_ylim(0, None)
        ax.grid(True, which="major", linewidth=0.3, color="0.85")
        panel(ax, "ab"[k])
    axs[0].legend(loc="lower right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig3_prefill_length")

def fig3():
    bw = R["bandwidth"]
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.75))
    for k, (q, ax) in enumerate(zip(("q8_0", "q4_k_m"), axs)):
        d = R["t_sweep"][q]
        t = d["t"]
        by = R["bytes"][q]
        roof = [bw["gbs"][bw["t"].index(x)] * 1e9 / by for x in t]
        ax.plot(t, roof, color="0.25", linestyle=":", linewidth=1.2, label="DRAM roofline")
        ax.fill_between(t, roof, [max(roof) * 1.3] * len(t), color="0.92", linewidth=0)
        ax.text(1.15, max(roof) * 0.99, "above measured\nDRAM bandwidth", fontsize=7, style="italic", color="0.35", va="top")
        gs = [gstat(v) for v in d["goose"]]
        ax.errorbar(t, [g[0] for g in gs], yerr=[[g[1] for g in gs], [g[2] for g in gs]], **G)
        ax.errorbar(t, [v[0] for v in d["llama_cpp"]], yerr=[v[1] for v in d["llama_cpp"]], linestyle="--", **L)
        g8 = gs[-1][0]
        ax.annotate(f"{100 * g8 / roof[-1]:.0f}% of bound", xy=(t[-1], g8), xytext=(t[-1] - 2.6, g8 * 0.62), fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", linewidth=0.5, color="0.3"))
        ax.set_xticks(t)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.set_xlim(0.6, t[-1] + 0.4)
        ax.set_ylim(0, max(roof) * 1.08)
        ax.set_xlabel("Threads (one per P-core)")
        if k == 0: ax.set_ylabel("Decode throughput (tokens s$^{-1}$)")
        ax.set_title(f"{NAMES[q]}  ({by / 1e9:.2f} GB of weights per token)", pad=4)
        ax.grid(True, which="major", linewidth=0.3, color="0.85")
        panel(ax, "ab"[k])
    axs[0].legend(loc="lower right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig2_decode_threads")

fig1()
if R.get("pp_sweep"): fig2()
if R.get("t_sweep") and R.get("bandwidth"): fig3()
if "--md" in sys.argv:
    print("| test | model | goose-llama (median [min, max]) | llama.cpp | ratio |\n|---|---|---|---|---|")
    for r in R["rows"]:
        g = r["goose_runs"]
        lc = f"{r['llama_cpp']:.1f} ± {r['llama_cpp_sd']:.1f}" if "llama_cpp_sd" in r else f"{r['llama_cpp']:.1f} [{min(r['llama_cpp_runs']):.1f}, {max(r['llama_cpp_runs']):.1f}]"
        print(f"| {r['test']} | {NAMES[r['model']]} | {r['goose']:.1f} [{min(g):.1f}, {max(g):.1f}] | {lc} | {r['goose'] / r['llama_cpp']:.2f}× |")
print("wrote", ", ".join(sorted(p.name for p in OUT.glob("fig*.svg"))))
