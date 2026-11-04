import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "llama.cpp" / "gguf-py"))
from gguf import GGUFReader
from gguf.quants import dequantize
r = GGUFReader(sys.argv[1])
want = dict(zip(sys.argv[2::2], map(int, sys.argv[3::2])))
for t in r.tensors:
    if t.name in want:
        row = int(want[t.name])
        d = dequantize(t.data, t.tensor_type).reshape(-1, int(t.shape[0]))[row].astype(np.float64)
        print(t.name, "ty=", int(t.tensor_type), "row=", row, "first=", d[:4].astype(np.float32).tolist(), "sum=", d.sum(), "sumsq=", (d*d).sum())
