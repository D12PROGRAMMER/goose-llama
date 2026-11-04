import std;
import os;

struct Go { n: i64 }
struct Rd { w: i64, t0: i64, t1: i64, s: u64 }

let SZ = 24 << 20;
let REPS = 6;

simd fn rd(a: const u64[:]) -> u64 {
    var s: u64[8] = [0; 8];
    for i in a.len / 8 { for k in 8 { s[k] ^= a[i * 8 + k]; } }
    return s[0] ^ s[1] ^ s[2] ^ s[3] ^ s[4] ^ s[5] ^ s[6] ^ s[7];
}

thread_fn reader(w: i64) {
    pin_thread(2 * w);
    var a: u64[>..] = [];
    a.resize(SZ, 0);
    for i in SZ { a[i] = i as u64 * 2654435761; }
    qput(Rd { w: w, t0: 0, t1: 0, s: 0 });
    let g = qget<Go>();
    let t0 = clock_ns();
    var s: u64 = 0;
    for r in REPS { s ^= rd(a); }
    qput(Rd { w: w, t0: t0, t1: clock_ns(), s: s + g.n as u64 });
}

fn main() {
    for n in 1..9 {
        var ids: i64[>..] = [];
        for w in n { ids.push(thread_spawn(reader, w)); }
        for w in n { qget<Rd>(); }
        for w in n { qput(Go { n: 0 }); }
        var a: i64 = 1 << 62;
        var b: i64 = 0;
        for w in n { let r = qget<Rd>(); a = min(a, r.t0); b = max(b, r.t1); }
        for id in ids { thread_wait(id); }
        print("threads ", n, " gbs ", (n * SZ * 8 * REPS) as f64 / (b - a) as f64);
    }
}
