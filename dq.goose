import llama;
fn main() {
    let a = args();
    var f: u8[>..] = [];
    guard read_file(a[1], f) else { abort("read"); }
    var kvs: KV[>..] = [];
    var ts: TI[>..] = [];
    let base = parse(f, kvs, ts);
    for k in (a.len - 2) / 2 {
        let i = tfind(f, ts, a[2 + k * 2]);
        let r = parse_int(a[3 + k * 2]);
        let t = ts[i];
        var y: f32[>..] = [];
        deq(f, t.ty, base + t.off + r * rbytes(t.ty, t.ne0), t.ne0, y);
        var s: f64 = 0;
        var s2: f64 = 0;
        for v in y { s += v; s2 += v * v; }
        print(a[2 + k * 2], " ty=", t.ty, " row=", r, " first=", y[..4], " sum=", s, " sumsq=", s2);
    }
}
