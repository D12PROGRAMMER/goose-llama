import subprocess, sys
sys.stdout.reconfigure(encoding="utf-8")
A = __import__("pathlib").Path(__file__).resolve().parents[2]
M = str(A / "models" / "qwen2.5-1.5b-instruct-q8_0.gguf")
LS = str(A / "llama.cpp" / "build" / "bin" / "llama-simple.exe")
GL = str(A / "goose-llama" / "llama.exe")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
T = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
cases = [T.format("Write a Python function that checks if a number is prime."), T.format("Explain why the sky is blue in three sentences."),
         "Once upon a time, in a small village by the sea,", T.format("List five prime numbers greater than 100.")]
for c in cases:
    ref = subprocess.run([LS, "-m", M, "-n", str(N), "-ngl", "0", c], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    ref = ref[len(c):] if ref.startswith(c) else ref
    got = subprocess.run([GL, "-m", M, "-p", c, "-n", str(N), "--draft", "0"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    got = "\n".join(l for l in got.split("\n") if not l.startswith(("qwen2 L=", "qwen3 L=", "llama L=", "native k-quant")))
    ref = ref.split("<|im_end|>")[0].rstrip()
    got = got.rstrip()
    k = next((i for i in range(min(len(ref), len(got))) if ref[i] != got[i]), min(len(ref), len(got)))
    print(f"{'SAME' if ref == got else 'DIFF'} common_prefix={k} ref_len={len(ref)} got_len={len(got)}")
    if ref != got: print("  ref:", repr(ref[max(0,k-40):k+60])); print("  got:", repr(got[max(0,k-40):k+60]))
