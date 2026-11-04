import subprocess, sys, re, os, tempfile
sys.stdout.reconfigure(encoding="utf-8")
A = __import__("pathlib").Path(__file__).resolve().parents[2]
M = str(A / "models" / "qwen2.5-1.5b-instruct-q8_0.gguf")
LT = str(A / "llama.cpp" / "build" / "bin" / "llama-tokenize.exe")
GL = str(A / "goose-llama" / "llama.exe")
cases = [
 "Hello world", "I'm sure they'll say we've done it, DON'T you think?",
 "1234567 + 89 = 1234656", "  leading   spaces\n\n\ttabs\r\nand CRLF  ",
 "naïve café über Straße — “quotes” … 日本語のテキスト 中文 한국어",
 "emoji 🦆🔥👍🏽 mixed", "def f(x):\n    return x**2  # comment\n\n\n",
 "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", "a\u00a0b\u3000c", "!!! ??? ... ---", "x'S y'Re",
 "Ελληνικά, русский язык, العربية, עברית, हिन्दी", "price: $1,299.99 (20% off!) 2024-11-03T12:00Z",
 "{\"key\": [1, 2, {\"n\": null}]}\n", "    \n  \n x", "über-cool ÆØÅ æøå ﬁ ½ ²³",
]
d = tempfile.mkdtemp()
bad = 0
for i, c in enumerate(cases):
    fp = os.path.join(d, f"{i}.txt")
    open(fp, "wb").write(c.encode())
    ref = subprocess.run([LT, "-m", M, "-f", fp, "--ids", "--log-disable"], capture_output=True, text=True, encoding="utf-8").stdout.strip()
    ref = [int(x) for x in re.findall(r"-?\d+", ref.splitlines()[-1])] if ref else None
    out = subprocess.run([GL, "-m", M, "-f", fp, "-n", "0", "-t", "1", "-v"], capture_output=True, text=True, encoding="utf-8").stdout
    got = [int(x) for x in re.findall(r"-?\d+", next(l for l in out.splitlines() if l.startswith("tokens:")))]
    ok = ref == got
    bad += not ok
    print("OK " if ok else "BAD", repr(c)[:70], "" if ok else f"\n  ref {ref}\n  got {got}")
print("mismatches:", bad, "/", len(cases))
