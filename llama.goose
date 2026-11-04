import std;
import os;
import math;
import binary;
import dictionary;
import cuda;

struct Q8 { d: u16, q: i8[32] }
struct QU { d: u16, q: u8[32] }
struct XC { c: i32[8], d: f32[8] }
struct K4 { d: u16, dm: u16, sc: u8[8], m: u8[8], q: u8[128] }
struct K6 { d: u16, sc: i8[16], ql: u8[128], qh: u8[64] }
struct XK { q: Q8[8], s: i16[16] }
struct Sh { a: const QU[:], b: const K4[:], c: const K6[:], fm: const i32[:], off: const i64[:] }
struct Cfg { L: i64, d: i64, ff: i64, nh: i64, nkv: i64, hd: i64, V: i64, nw: i64, ctx: i64, sp: i64, fg: i64, nb: i64, fb: bool, pin: bool, neox: bool, qkn: bool, eps: f32, base: f32 }
struct J<T> { op: i32, l: i32, pos: i32, nt: i32, k: i32, tr: i32, nc: i32, tag: T, x: f32[] }
struct St { w: i32, busy: i64, n: i64, ex: i64, ta: i64, tf: i64, tl: i64, tr: i64[] }
struct Res { w: i32, y: f32[] }
struct P<T> { src: i32, rd: i32, tag: T, y: f32[] }
struct KVm<T> { src: i32, l: i32, h: i32, s0: i32, n: i32, tag: T, y: f32[] }
struct Geo { ha: i64, hb: i64, ka: i64, kb: i64, kr: i64, fa: i64, fb: i64, va: i64, vb: i64 }
struct KV { ko: i64, kl: i64, ty: i64, vo: i64 }
struct TI { no: i64, nl: i64, ty: i64, ne0: i64, ne1: i64, off: i64 }

var C: Cfg;
var HT: f32[65536] = [0.0; 65536];

fn lo(r: i64, w: i64, n: i64) -> i64 { r * w / n }
fn hlo(w: i64) -> i64 {
    if w >= C.nw { return C.nh; }
    if C.nw < C.nkv || C.nh % C.nkv != 0 { return w * (C.nh / C.nw) + min(w, C.nh % C.nw); }
    let gs = C.nh / C.nkv;
    var g = 0;
    while lo(C.nw, g + 1, C.nkv) <= w { g++; }
    let a = lo(C.nw, g, C.nkv);
    let m = lo(C.nw, g + 1, C.nkv) - a;
    let i = w - a;
    return g * gs + i * (gs / m) + min(i, gs % m);
}
fn kab(w: i64) -> i64, i64 {
    let ha = hlo(w);
    let hb = hlo(w + 1);
    let g = C.nh / C.nkv;
    return ha / g, if hb > ha { (hb - 1) / g + 1 } else { ha / g };
}
fn kvm(h: i64, w: i64) -> i64, i64 {
    var i = 0;
    var n = 0;
    for v in C.nw { let a, b = kab(v); if a <= h && h < b { if v < w { i++; } n++; } }
    return i, n;
}
fn nhd(w: i64) -> i64 { hlo(w + 1) - hlo(w) }
fn ksl(h: i64, w: i64) -> i64, i64 {
    var hm = 0;
    for v in C.nw { let a, b = kab(v); if a <= h && h < b { hm = max(hm, nhd(v)); } }
    var tot = 0;
    var bef = 0;
    var me = 0;
    for v in C.nw { let a, b = kab(v); if a <= h && h < b { let u = hm - nhd(v); tot += u; if v < w { bef += u; } if v == w { me = u; } } }
    if tot == 0 { let i, n = kvm(h, w); return lo(C.hd, i, n), lo(C.hd, i + 1, n); }
    return C.hd * bef / tot, C.hd * (bef + me) / tot;
}
fn geo(w: i64) -> Geo {
    let ha = hlo(w);
    let hb = hlo(w + 1);
    let ka, kb = kab(w);
    var kr = 0;
    for h in ka..kb { let s0, s1 = ksl(h, w); kr += s1 - s0; }
    return Geo { ha, hb, ka, kb, kr, lo(C.ff / 32 / C.fg, w, C.nw) * C.fg, lo(C.ff / 32 / C.fg, w + 1, C.nw) * C.fg, lo(C.V, w, C.nw), lo(C.V, w + 1, C.nw) };
}
fn tsz(g: Geo, t: i64) -> i64, i64 {
    let hd = C.hd;
    if t == 0 { return (g.hb - g.ha) * hd, C.d / 32; }
    if t < 3 { return g.kr, C.d / 32; }
    if t == 3 { return C.d, (g.hb - g.ha) * hd / 32; }
    if t < 6 { return (g.fb - g.fa) * 32, C.d / 32; }
    if t == 6 { return C.d, g.fb - g.fa; }
    return g.vb - g.va, C.d / 32;
}
fn tdims(k: i64) -> i64, i64 {
    if k == C.L * 7 { return C.V, C.d; }
    let t = k % 7;
    if t == 0 { return C.nh * C.hd, C.d; }
    if t < 3 { return C.nkv * C.hd, C.d; }
    if t == 3 { return C.d, C.nh * C.hd; }
    if t < 6 { return C.ff, C.d; }
    return C.d, C.ff;
}
fn mmk(s: Sh, g: Geo, l: i64, t: i64, x: const f32[:], xq: const Q8[:], xp: const XK[:], nt: i64, o: f32[:]) {
    let k = if t == 7 { C.L * 7 } else { l * 7 + t };
    let r, nb = tsz(g, t);
    let p = s.off[k];
    if s.fm[k] == 2 { mm6(s.c[p..p + r * nb / 8], xp, nt, o); return; }
    if s.fm[k] == 1 { mm4(s.b[p..p + r * nb / 8], xp, nt, o); return; }
    if nt == 1 || !C.fb { mm(s.a[p..p + r * nb], xq, nt, o); }
    else { mm8b(s.a[p..p + r * nb], xq, nt, o); }
}
simd fn qp(x: const f32[:], xp: XK[>..]&) {
    for c in x.len / 256 {
        let v = x[c * 256..c * 256 + 256];
        var m: f32 = 0;
        for e in v { m = max(m, abs(e)); }
        let id: f32 = if m != 0 { 127 / m } else { 0 };
        let h = f2h(m / 127);
        let r .= xp.push(XK { q: [Q8 { d: h, q: [0; 32] }; 8], s: [0; 16] });
        for g in 8 {
            for p in 32 {
                let b = (p >> 2) + 8 * (g & 1);
                let q = round_even(v[16 * b + 4 * (g >> 1) + (p & 3)] * id) as i8;
                r.q[g].q[p] = q;
                r.s[b] += q;
            }
        }
    }
}
fn qx(s: Sh, k0: i64, k1: i64, x: const f32[:], xq: Q8[>..]&, xp: XK[>..]&) {
    var n0 = false;
    var n1 = false;
    for k in k0..k1 { if s.fm[k] == 0 { n0 = true; } else { n1 = true; } }
    if n0 { qrow(x, xq); }
    if n1 { qp(x, xp); }
}
inline fn k4p(se: i32[32]&, so: i32[32]&, t: i64, lo: const u8[32], hi: const u8[32], xa: const i8[32], xb: const i8[32]) {
    var a: i32[8] = [0; 8];
    var b: i32[8] = [0; 8];
    for k in 8 { a[k] = se[t * 8 + k]; b[k] = so[t * 8 + k]; }
    let u = dot_quads_i(lo, xa, a);
    let v = dot_quads_i(hi, xb, b);
    for k in 8 { se[t * 8 + k] = u[k]; so[t * 8 + k] = v[k]; }
}
inline fn k6p(se: i32[32]&, so: i32[32]&, t: i64, u0: const u8[32], u1: const u8[32], u2: const u8[32], u3: const u8[32], x0: const i8[32], x1: const i8[32], x2: const i8[32], x3: const i8[32]) {
    var a: i32[8] = [0; 8];
    var b: i32[8] = [0; 8];
    for k in 8 { a[k] = se[t * 8 + k]; b[k] = so[t * 8 + k]; }
    let a1 = dot_quads_i(u2, x2, dot_quads_i(u0, x0, a));
    let b1 = dot_quads_i(u3, x3, dot_quads_i(u1, x1, b));
    for k in 8 { se[t * 8 + k] = a1[k]; so[t * 8 + k] = b1[k]; }
}
inline fn k4f(acc: f32[32]&, t: i64, se: const i32[32], so: const i32[32], fe: const f32[8], fo: const f32[8], bv: const i16[16], m16: const i16[16], dm: f32, dx: f32) {
    let mz = madd16(bv, m16);
    let fm = dm * dx;
    for k in 8 { acc[t * 8 + k] += se[t * 8 + k] * (fe[k] * dx) + so[t * 8 + k] * (fo[k] * dx) - mz[k] * fm; }
}
inline fn k6f(acc: f32[32]&, t: i64, se: const i32[32], so: const i32[32], fe: const f32[8], fo: const f32[8], bv: const i16[16], s16: const i16[16], d: f32, dx: f32) {
    let mz = madd16(bv, s16);
    let f0 = 32 * d * dx;
    for k in 8 { acc[t * 8 + k] += se[t * 8 + k] * (fe[k] * dx) + so[t * 8 + k] * (fo[k] * dx) - mz[k] * f0; }
}
simd fn mm4(a: const K4[:], x: const XK[:], nt: i64, o: f32[:]) {
    let ns = x.len / nt;
    let R = a.len / ns;
    var c0 = 0;
    while c0 < nt {
        let c1 = min(c0 + 16, nt);
        for r in R {
            let row = a[r * ns..(r + 1) * ns];
            a.prefetch((r + 1) * ns + 28, ns);
            var t0 = c0;
            while t0 < c1 {
                let tn = min(4, c1 - t0);
                let j0 = t0;
                let j1 = min(t0 + 1, c1 - 1);
                let j2 = min(t0 + 2, c1 - 1);
                let j3 = min(t0 + 3, c1 - 1);
                let x0 = x[j0 * ns..j0 * ns + ns];
                let x1 = x[j1 * ns..j1 * ns + ns];
                let x2 = x[j2 * ns..j2 * ns + ns];
                let x3 = x[j3 * ns..j3 * ns + ns];
                assert(row.len == ns && x0.len == ns && x1.len == ns && x2.len == ns && x3.len == ns);
                var acc: f32[32] = [0.0; 32];
                for &e, sb in row {
                    var m16: i16[16] = [0; 16];
                    for i in 16 { m16[i] = e.m[i >> 1]; }
                    let d = HT[e.d];
                    var fe: f32[8] = [0.0; 8];
                    var fo: f32[8] = [0.0; 8];
                    for k in 8 { fe[k] = d * e.sc[k >> 1]; fo[k] = d * e.sc[4 + (k >> 1)]; }
                    var se: i32[32] = [0; 32];
                    var so: i32[32] = [0; 32];
                    for c in 4 {
                        var lo: u8[32] = [0; 32];
                        var hi: u8[32] = [0; 32];
                        for l in 32 { let v = e.q[c * 32 + l]; lo[l] = v & 15; hi[l] = v >> 4; }
                        k4p(se, so, 0, lo, hi, x0[sb].q[c * 2].q, x0[sb].q[c * 2 + 1].q);
                        if tn > 1 { k4p(se, so, 1, lo, hi, x1[sb].q[c * 2].q, x1[sb].q[c * 2 + 1].q); }
                        if tn > 2 { k4p(se, so, 2, lo, hi, x2[sb].q[c * 2].q, x2[sb].q[c * 2 + 1].q); }
                        if tn > 3 { k4p(se, so, 3, lo, hi, x3[sb].q[c * 2].q, x3[sb].q[c * 2 + 1].q); }
                    }
                    let dm = HT[e.dm];
                    k4f(acc, 0, se, so, fe, fo, x0[sb].s, m16, dm, HT[x0[sb].q[0].d]);
                    if tn > 1 { k4f(acc, 1, se, so, fe, fo, x1[sb].s, m16, dm, HT[x1[sb].q[0].d]); }
                    if tn > 2 { k4f(acc, 2, se, so, fe, fo, x2[sb].s, m16, dm, HT[x2[sb].q[0].d]); }
                    if tn > 3 { k4f(acc, 3, se, so, fe, fo, x3[sb].s, m16, dm, HT[x3[sb].q[0].d]); }
                }
                for t in tn { let q = t * 8; o[(t0 + t) * R + r] = acc[q] + acc[q + 4] + (acc[q + 2] + acc[q + 6]) + (acc[q + 1] + acc[q + 5] + (acc[q + 3] + acc[q + 7])); }
                t0 += tn;
            }
        }
        c0 = c1;
    }
}
simd fn mm6(a: const K6[:], x: const XK[:], nt: i64, o: f32[:]) {
    let ns = x.len / nt;
    let R = a.len / ns;
    var c0 = 0;
    while c0 < nt {
        let c1 = min(c0 + 16, nt);
        for r in R {
            let row = a[r * ns..(r + 1) * ns];
            a.prefetch((r + 1) * ns + 20, ns);
            var t0 = c0;
            while t0 < c1 {
                let tn = min(4, c1 - t0);
                let j0 = t0;
                let j1 = min(t0 + 1, c1 - 1);
                let j2 = min(t0 + 2, c1 - 1);
                let j3 = min(t0 + 3, c1 - 1);
                let x0 = x[j0 * ns..j0 * ns + ns];
                let x1 = x[j1 * ns..j1 * ns + ns];
                let x2 = x[j2 * ns..j2 * ns + ns];
                let x3 = x[j3 * ns..j3 * ns + ns];
                assert(row.len == ns && x0.len == ns && x1.len == ns && x2.len == ns && x3.len == ns);
                var acc: f32[32] = [0.0; 32];
                for &e, sb in row {
                    var s16: i16[16] = [0; 16];
                    for i in 16 { s16[i] = e.sc[i]; }
                    let d = HT[e.d];
                    var fe: f32[8] = [0.0; 8];
                    var fo: f32[8] = [0.0; 8];
                    for k in 8 { fe[k] = d * e.sc[k]; fo[k] = d * e.sc[8 + k]; }
                    var se: i32[32] = [0; 32];
                    var so: i32[32] = [0; 32];
                    for h in 2 {
                        var u0: u8[32] = [0; 32];
                        var u1: u8[32] = [0; 32];
                        var u2: u8[32] = [0; 32];
                        var u3: u8[32] = [0; 32];
                        for l in 32 {
                            let p = e.ql[h * 64 + l];
                            let q = e.ql[h * 64 + 32 + l];
                            let c = e.qh[h * 32 + l];
                            u0[l] = (p & 15) | ((c & 3) << 4);
                            u1[l] = (q & 15) | ((c & 12) << 2);
                            u2[l] = (p >> 4) | (c & 48);
                            u3[l] = (q >> 4) | ((c >> 2) & 48);
                        }
                        k6p(se, so, 0, u0, u1, u2, u3, x0[sb].q[h * 4].q, x0[sb].q[h * 4 + 1].q, x0[sb].q[h * 4 + 2].q, x0[sb].q[h * 4 + 3].q);
                        if tn > 1 { k6p(se, so, 1, u0, u1, u2, u3, x1[sb].q[h * 4].q, x1[sb].q[h * 4 + 1].q, x1[sb].q[h * 4 + 2].q, x1[sb].q[h * 4 + 3].q); }
                        if tn > 2 { k6p(se, so, 2, u0, u1, u2, u3, x2[sb].q[h * 4].q, x2[sb].q[h * 4 + 1].q, x2[sb].q[h * 4 + 2].q, x2[sb].q[h * 4 + 3].q); }
                        if tn > 3 { k6p(se, so, 3, u0, u1, u2, u3, x3[sb].q[h * 4].q, x3[sb].q[h * 4 + 1].q, x3[sb].q[h * 4 + 2].q, x3[sb].q[h * 4 + 3].q); }
                    }
                    k6f(acc, 0, se, so, fe, fo, x0[sb].s, s16, d, HT[x0[sb].q[0].d]);
                    if tn > 1 { k6f(acc, 1, se, so, fe, fo, x1[sb].s, s16, d, HT[x1[sb].q[0].d]); }
                    if tn > 2 { k6f(acc, 2, se, so, fe, fo, x2[sb].s, s16, d, HT[x2[sb].q[0].d]); }
                    if tn > 3 { k6f(acc, 3, se, so, fe, fo, x3[sb].s, s16, d, HT[x3[sb].q[0].d]); }
                }
                for t in tn { let q = t * 8; o[(t0 + t) * R + r] = acc[q] + acc[q + 4] + (acc[q + 2] + acc[q + 6]) + (acc[q + 1] + acc[q + 5] + (acc[q + 3] + acc[q + 7])); }
                t0 += tn;
            }
        }
        c0 = c1;
    }
}
fn xcor(x: const Q8[:]) -> XC[>..] {
    var o: XC[>..] = [];
    let k: u8[32] = [128; 32];
    for e in x { let r .= o.push(XC { c: dot_bytes(k, e.q), d: [HT[e.d]; 8] }); for i in 8 { r.c[i] = -r.c[i]; } }
    return o;
}
inline fn u8p(p0: f32[32]&, p1: f32[32]&, t: i64, wa: const u8[32], wb: const u8[32], xq: const i8[32], xc: const XC, da: f32, db: f32) {
    let v = dot_bytes(wa, xq, xc.c);
    let u = dot_bytes(wb, xq, xc.c);
    for k in 8 { p0[t * 8 + k] = mul_add(v[k] as f32, xc.d[k] * da, p0[t * 8 + k]); p1[t * 8 + k] = mul_add(u[k] as f32, xc.d[k] * db, p1[t * 8 + k]); }
}
simd fn mm8b(a: const QU[:], x: const Q8[:], nt: i64, o: f32[:]) {
    let nb = x.len / nt;
    let R = a.len / nb;
    let xc = xcor(x);
    var c0 = 0;
    while c0 < nt {
        let c1 = min(c0 + 16, nt);
        var r = 0;
        while r < R {
            let r2 = min(r + 1, R - 1);
            let ra = a[r * nb..(r + 1) * nb];
            let rb = a[r2 * nb..(r2 + 1) * nb];
            a.prefetch((r + 2) * nb + 120, 2 * nb);
            var t0 = c0;
            while t0 < c1 {
                let j0 = min(t0, c1 - 1) * nb;
                let j1 = min(t0 + 1, c1 - 1) * nb;
                let j2 = min(t0 + 2, c1 - 1) * nb;
                let j3 = min(t0 + 3, c1 - 1) * nb;
                let x0 = x[j0..j0 + nb];
                let x1 = x[j1..j1 + nb];
                let x2 = x[j2..j2 + nb];
                let x3 = x[j3..j3 + nb];
                let e0 = xc[j0..j0 + nb];
                let e1 = xc[j1..j1 + nb];
                let e2 = xc[j2..j2 + nb];
                let e3 = xc[j3..j3 + nb];
                assert(ra.len == nb && rb.len == nb && x0.len == nb && x1.len == nb && x2.len == nb && x3.len == nb && e0.len == nb && e1.len == nb && e2.len == nb && e3.len == nb);
                var p0: f32[32] = [0.0; 32];
                var p1: f32[32] = [0.0; 32];
                for b in nb {
                    let da = HT[ra[b].d];
                    let db = HT[rb[b].d];
                    u8p(p0, p1, 0, ra[b].q, rb[b].q, x0[b].q, e0[b], da, db);
                    u8p(p0, p1, 1, ra[b].q, rb[b].q, x1[b].q, e1[b], da, db);
                    u8p(p0, p1, 2, ra[b].q, rb[b].q, x2[b].q, e2[b], da, db);
                    u8p(p0, p1, 3, ra[b].q, rb[b].q, x3[b].q, e3[b], da, db);
                }
                for t in min(4, c1 - t0) {
                    let q = t * 8;
                    o[(t0 + t) * R + r] = p0[q] + p0[q + 4] + (p0[q + 2] + p0[q + 6]) + (p0[q + 1] + p0[q + 5] + (p0[q + 3] + p0[q + 7]));
                    if r + 1 < R { o[(t0 + t) * R + r + 1] = p1[q] + p1[q + 4] + (p1[q + 2] + p1[q + 6]) + (p1[q + 1] + p1[q + 5] + (p1[q + 3] + p1[q + 7])); }
                }
                t0 += 4;
            }
            r += 2;
        }
        c0 = c1;
    }
}
simd fn mm(a: const QU[:], x: const Q8[:], nt: i64, o: f32[:]) {
    let nb = x.len / nt;
    let R = a.len / nb;
    let xc = xcor(x);
    for r in R {
        let row = a[r * nb..(r + 1) * nb];
        a.prefetch((r + 1) * nb + 120, nb);
        for t in nt {
            let xr = x[t * nb..(t + 1) * nb];
            let cr = xc[t * nb..(t + 1) * nb];
            var a0: f32[8] = [0.0; 8];
            var a1: f32[8] = [0.0; 8];
            var b = 0;
            while b + 1 < nb {
                let v = dot_bytes(row[b].q, xr[b].q, cr[b].c);
                let u = dot_bytes(row[b + 1].q, xr[b + 1].q, cr[b + 1].c);
                let f = HT[row[b].d] * HT[xr[b].d];
                let g = HT[row[b + 1].d] * HT[xr[b + 1].d];
                for k in 8 { a0[k] += v[k] * f; a1[k] += u[k] * g; }
                b += 2;
            }
            if b < nb { let v = dot_bytes(row[b].q, xr[b].q, cr[b].c); let f = HT[row[b].d] * HT[xr[b].d]; for k in 8 { a0[k] += v[k] * f; } }
            let acc = a0 + a1;
            o[t * R + r] = acc[0] + acc[4] + (acc[2] + acc[6]) + (acc[1] + acc[5] + (acc[3] + acc[7]));
        }
    }
}
fn f2h(x: f32) -> u16 {
    let a: f32[1] = [x];
    let b = bytes_of(a);
    let u = b[0] as u32 | (b[1] as u32) << 8 | (b[2] as u32) << 16 | (b[3] as u32) << 24;
    let s = (u >> 16) & 32768;
    let e = ((u >> 23) & 255) as i32 - 112;
    let m = u & 8388607;
    if e >= 31 { return (s | 31744) as u16; }
    if e <= 0 {
        if e < -10 { return s as u16; }
        let mm = m | 8388608;
        let sh = (14 - e) as u32;
        var r = mm >> sh;
        let rem = mm & ((1 << sh) - 1);
        let hf: u32 = 1 << (sh - 1);
        if rem > hf || (rem == hf && (r & 1) != 0) { r++; }
        return (s | r) as u16;
    }
    var r = (e as u32 << 10) | (m >> 13);
    let rem = m & 8191;
    if rem > 4096 || (rem == 4096 && (r & 1) != 0) { r++; }
    return (s | r) as u16;
}
fn h2f(h: u16) -> f32 {
    let e = (h >> 10) & 31;
    let m = (h & 1023) as u32;
    let s = (h as u32 >> 15) << 31;
    if e == 0 {
        let v = m * 0.000000059604645;
        let r: f32 = if s != 0 { -v } else { v };
        return r;
    }
    if e == 31 { return binary::f32_of_bits(s | 2139095040 | (m << 13)); }
    return binary::f32_of_bits(s | ((e as u32 + 112) << 23) | (m << 13));
}
simd fn qrow(x: const f32[:], o: Q8[>..]&) {
    for b in x.len / 32 {
        let v = x[b * 32..b * 32 + 32];
        var m: f32 = 0;
        for e in v { m = max(m, abs(e)); }
        let id: f32 = if m != 0 { 127 / m } else { 0 };
        let r .= o.push(Q8 { d: f2h(m / 127), q: [0; 32] });
        for e, j in v { r.q[j] = round_even(e * id) as i8; }
    }
}
fn rms(x: const f32[:], w: const f32[:], y: f32[:]) {
    var s: f64 = 0;
    for v in x { s += v * v; }
    let mean = (s / x.len) as f32;
    let sc: f32 = 1 / sqrt(mean + C.eps);
    for i in x.len { y[i] = x[i] * sc * w[i]; }
}
fn rtab(rt: f32[>..]&, n: i64) {
    let hd = C.hd;
    let ts = pow(C.base as f64, -2.0 / hd) as f32;
    while rt.len < n * hd {
        var th: f32 = rt.len / hd;
        for i in hd / 2 { rt.push(cos(th)); rt.push(sin(th)); th *= ts; }
    }
}
fn rope(v: f32[:], rt: const f32[:], pos: i64) {
    let hd = C.hd;
    let h2 = hd / 2;
    let r = rt[pos * hd..(pos + 1) * hd];
    for i in h2 {
        let cs = r[2 * i];
        let sn = r[2 * i + 1];
        for h in v.len / hd {
            let o = h * hd;
            let ia = if C.neox { o + i } else { o + 2 * i };
            let ib = if C.neox { o + i + h2 } else { o + 2 * i + 1 };
            let a = v[ia];
            let b = v[ib];
            v[ia] = a * cs - b * sn;
            v[ib] = a * sn + b * cs;
        }
    }
}
fn hnorm(v: f32[:], w: const f32[:]) {
    let hd = C.hd;
    for h in v.len / hd { let s = v[h * hd..(h + 1) * hd]; rms(s, w, s); }
}

fn dot8(a: const f32[:], b: const f32[:]) -> f32 {
    var c: f32[8] = [0.0; 8];
    for j in a.len / 8 { for k in 8 { c[k] = mul_add(a[j * 8 + k], b[j * 8 + k], c[k]); } }
    return c[0] + c[4] + (c[2] + c[6]) + (c[1] + c[5] + (c[3] + c[7]));
}
fn attb<T>(w: i64, g: Geo, s: Sh, nv: const f32[:], kc: f32[:], vc: f32[:], l: i64, pos: i64, nt: i64, x: const f32[:], par: const i64[:], dep: const i64[:], rt: const f32[:]) -> f32[>..] {
    let tree = par.len > 0;
    let hd = C.hd;
    let nq = (g.hb - g.ha) * hd;
    let nk = (g.kb - g.ka) * hd;
    var out: f32[>..] = [];
    out.resize(nt * C.d, 0);
    if nq == 0 { return out; }
    var q: f32[>..] = [];
    q.resize(nt * nq, 0);
    var kk: f32[>..] = [];
    kk.resize(nt * nk, 0);
    var vv: f32[>..] = [];
    vv.resize(nt * nk, 0);
    var xq: Q8[>..] = [];
    var xp: XK[>..] = [];
    qx(s, l * 7, l * 7 + 3, x, xq, xp);
    let kr = g.kr;
    var kp: f32[>..] = [];
    kp.resize(nt * kr, 0);
    var vp: f32[>..] = [];
    vp.resize(nt * kr, 0);
    mmk(s, g, l, 1, x, xq, xp, nt, kp);
    mmk(s, g, l, 2, x, xq, xp, nt, vp);
    var o = 0;
    var need = 0;
    for h in g.ka..g.kb {
        let s0, s1 = ksl(h, w);
        let n = s1 - s0;
        let c = (h - g.ka) * hd + s0;
        var y: f32[>..] = [];
        for t in nt { y.append(kp[t * kr + o..t * kr + o + n]); }
        for t in nt { y.append(vp[t * kr + o..t * kr + o + n]); }
        for v in C.nw {
            let a, b = kab(v);
            if v != w && a <= h && h < b {
                let u0, u1 = ksl(h, v);
                if u1 > u0 { need++; }
                if n > 0 { const for <K> in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] { if K == v + 1 { qput(KVm<u8[K]> { src: w as i32, l: l as i32, h: h as i32, s0: s0 as i32, n: n as i32, tag: [0; K], y: copy(y) }); } } }
            }
        }
        for t in nt { for j in n { kk[t * nk + c + j] = kp[t * kr + o + j]; vv[t * nk + c + j] = vp[t * kr + o + j]; } }
        o += n;
    }
    mmk(s, g, l, 0, x, xq, xp, nt, q);
    while need > 0 {
        let m = qget<KVm<T>>();
        let n = m.n as i64;
        let c = (m.h as i64 - g.ka) * hd + m.s0;
        for t in nt { for j in n { kk[t * nk + c + j] = m.y[t * n + j]; vv[t * nk + c + j] = m.y[(nt + t) * n + j]; } }
        need--;
    }
    let nb = l * (nq + 2 * nk + 2 * hd);
    for t in nt {
        let p = if tree { pos + dep[t] } else { pos + t };
        let qt = q[t * nq..(t + 1) * nq];
        let kt = kk[t * nk..(t + 1) * nk];
        let vt = vv[t * nk..(t + 1) * nk];
        for i in nq { qt[i] += nv[nb + i]; }
        for i in nk { kt[i] += nv[nb + nq + i]; vt[i] += nv[nb + nq + nk + i]; }
        if C.qkn {
            let qn = nb + nq + 2 * nk;
            hnorm(qt, nv[qn..qn + hd]);
            hnorm(kt, nv[qn + hd..qn + 2 * hd]);
        }
        rope(qt, rt, p);
        rope(kt, rt, p);
        let co = (l * C.ctx + pos + t) * nk;
        for i in nk { kc[co + i] = kt[i]; vc[co + i] = vt[i]; }
    }
    var att: f32[>..] = [];
    att.resize(nt * nq, 0);
    var sc: f32[>..] = [];
    sc.resize(pos + nt, 0);
    let gq = C.nh / C.nkv;
    let s0: f32 = 1 / sqrt(hd as f32);
    var an: i64[>..<] = [];
    for t in nt {
        let p = pos + t;
        if tree {
            an.clear();
            for s in pos { an.push(s); }
            var u = t;
            var anc: i64[>..<] = [];
            while u >= 0 { anc.push(pos + u); u = par[u]; }
            for i in anc.len { an.push(anc[anc.len - 1 - i]); }
            for h in g.hb - g.ha {
                let qh = q[t * nq + h * hd..t * nq + (h + 1) * hd];
                let kh = (g.ha + h) / gq - g.ka;
                var mx: f32 = -1e30;
                for sl, i in an {
                    let ko = (l * C.ctx + sl) * nk + kh * hd;
                    sc[i] = dot8(qh, kc[ko..ko + hd]) * s0;
                    mx = max(mx, sc[i]);
                }
                var sum: f32 = 0;
                for i in an.len { sc[i] = exp(sc[i] - mx); sum += sc[i]; }
                let o = att[t * nq + h * hd..t * nq + (h + 1) * hd];
                for sl, i in an {
                    let vo = (l * C.ctx + sl) * nk + kh * hd;
                    let vr = vc[vo..vo + hd];
                    let pr = sc[i] / sum;
                    for e in hd { o[e] += pr * vr[e]; }
                }
            }
            continue;
        }
        for h in g.hb - g.ha {
            let qh = q[t * nq + h * hd..t * nq + (h + 1) * hd];
            let kh = (g.ha + h) / gq - g.ka;
            var mx: f32 = -1e30;
            for s in p + 1 {
                let ko = (l * C.ctx + s) * nk + kh * hd;
                let kr = kc[ko..ko + hd];
                sc[s] = dot8(qh, kr) * s0;
                mx = max(mx, sc[s]);
            }
            var sum: f32 = 0;
            for s in p + 1 { sc[s] = exp(sc[s] - mx); sum += sc[s]; }
            let o = att[t * nq + h * hd..t * nq + (h + 1) * hd];
            for s in p + 1 {
                let vo = (l * C.ctx + s) * nk + kh * hd;
                let vr = vc[vo..vo + hd];
                let pr = sc[s] / sum;
                for i in hd { o[i] += pr * vr[i]; }
            }
        }
    }
    var aq: Q8[>..] = [];
    var ap: XK[>..] = [];
    qx(s, l * 7 + 3, l * 7 + 4, att, aq, ap);
    mmk(s, g, l, 3, att, aq, ap, nt, out);
    return out;
}
fn ffnb(g: Geo, s: Sh, l: i64, nt: i64, x: const f32[:]) -> f32[>..] {
    var out: f32[>..] = [];
    out.resize(nt * C.d, 0);
    let nf = (g.fb - g.fa) * 32;
    if nf == 0 { return out; }
    var gg: f32[>..] = [];
    gg.resize(nt * nf, 0);
    var uu: f32[>..] = [];
    uu.resize(nt * nf, 0);
    var xq: Q8[>..] = [];
    var xp: XK[>..] = [];
    qx(s, l * 7 + 4, l * 7 + 6, x, xq, xp);
    mmk(s, g, l, 4, x, xq, xp, nt, gg);
    mmk(s, g, l, 5, x, xq, xp, nt, uu);
    for i in gg.len { let v = gg[i]; gg[i] = v / (1 + exp(-v)) * uu[i]; }
    var hq: Q8[>..] = [];
    var hp: XK[>..] = [];
    qx(s, l * 7 + 6, l * 7 + 7, gg, hq, hp);
    mmk(s, g, l, 6, gg, hq, hp, nt, out);
    return out;
}
fn lmb(g: Geo, s: Sh, nt: i64, x: const f32[:]) -> f32[>..] {
    var y: f32[>..] = [];
    y.resize(nt * (g.vb - g.va), 0);
    var xq: Q8[>..] = [];
    var xp: XK[>..] = [];
    qx(s, C.L * 7, C.L * 7 + 1, x, xq, xp);
    mmk(s, g, 0, 7, x, xq, xp, nt, y);
    return y;
}
fn topk(y: const f32[:], va: i64, k: i64) -> f32[>..] {
    var ix: i64[>..<] = [];
    for v, i in y {
        if ix.len < k || v > y[ix[ix.len - 1]] {
            var j = ix.len;
            while j > 0 && y[ix[j - 1]] < v { j--; }
            insert_at(ix, j, i);
            if ix.len > k { ix.pop(); }
        }
    }
    var o: f32[>..] = [];
    for i in ix { o.push(y[i]); o.push((va + i) as f32); }
    return o;
}
fn topks(y: const f32[:], va: i64, k: i64, nt: i64) -> f32[>..] {
    var o: f32[>..] = [];
    let n = y.len / nt;
    for t in nt { o.append(topk(y[t * n..(t + 1) * n], va, k)); }
    return o;
}
fn rx<T>(sp: i64) -> T { qget<T>() }
fn rxs<T>(w: i64, tg: i64, pb: f32[:], pn: i64[:], o: f32[:], L: i64, seg: bool) {
    let S = pb.len / C.nw;
    let n = o.len;
    var got = 0;
    for s in C.nw {
        if s != w && pn[s] == tg {
            let b = if seg { lo(n, s, C.nw) } else { s * L };
            for j in lo(n, s + 1, C.nw) - lo(n, s, C.nw) { o[b + j] = pb[s * S + j]; }
            pn[s] = -1;
            got++;
        }
    }
    while got < C.nw - 1 {
        let m = rx<P<T>>(C.sp);
        let s = m.src as i64;
        if m.rd as i64 == tg {
            let b = if seg { lo(n, s, C.nw) } else { s * L };
            for v, j in m.y { o[b + j] = v; }
            got++;
        }
        else { for v, j in m.y { pb[s * S + j] = v; } pn[s] = m.rd; }
    }
}
simd fn sumw(all: f32[:], n: i64, X: f32[:]) {
    let a = all[..n];
    for s in 1..C.nw { let q = all[s * n..(s + 1) * n]; for j in n { a[j] += q[j]; } }
    for j in n { X[j] += a[j]; }
}
fn pft(s: Sh, k: i64) {
    let p = s.off[k];
    if s.fm[k] == 0 { s.a.prefetch(p, PFB / 34); }
    else if s.fm[k] == 1 { s.b.prefetch(p, PFB / 148); }
    else { s.c.prefetch(p, PFB / 210); }
}
fn ar<T>(w: i64, p: f32[>..]&, rd: i64, X: f32[:], all: f32[:], pb: f32[:], pn: i64[:], tt: i64[>..]&, s: Sh, nk: i64) {
    let n = p.len;
    if tt.len < 60000 { tt.push(clock_ns()); }
    if n < C.d * 16 {
        const for <K> in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] { if K <= C.nw && K != w + 1 { qput(P<u8[K]> { src: w as i32, rd: (rd * 2) as i32, tag: [0; K], y: copy(p) }); } }
        if PFB > 0 { pft(s, nk); }
        if tt.len < 60000 { tt.push(clock_ns()); }
        for j in n { all[w * n + j] = p[j]; }
        rxs<T>(w, rd * 2, pb, pn, all[..C.nw * n], n, false);
        if tt.len < 60000 { tt.push(clock_ns()); }
        sumw(all, n, X);
        if tt.len < 60000 { tt.push(clock_ns()); }
        return;
    }
    let a0 = lo(n, w, C.nw);
    let L = lo(n, w + 1, C.nw) - a0;
    const for <K> in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] { if K <= C.nw && K != w + 1 { qput(P<u8[K]> { src: w as i32, rd: (rd * 2) as i32, tag: [0; K], y: p[lo(n, K - 1, C.nw)..lo(n, K, C.nw)] }); } }
    for j in L { all[w * L + j] = p[a0 + j]; }
    if tt.len < 60000 { tt.push(clock_ns()); }
    rxs<T>(w, rd * 2, pb, pn, all[..C.nw * L], L, false);
    sumw(all, L, X[a0..a0 + L]);
    if tt.len < 60000 { tt.push(clock_ns()); }
    const for <K> in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] { if K <= C.nw && K != w + 1 { qput(P<u8[K]> { src: w as i32, rd: (rd * 2 + 1) as i32, tag: [0; K], y: X[a0..a0 + L] }); } }
    rxs<T>(w, rd * 2 + 1, pb, pn, X[..n], 0, true);
    if tt.len < 60000 { tt.push(clock_ns()); }
}
fn wloop<T>(w: i64, s8: const QU[:], s4: const K4[:], s6: const K6[:], fm: const i32[:], nv: const f32[:]) {
    if C.pin { pin_thread(2 * w); }
    let g = geo(w);
    var off: i64[>..] = [];
    var o: i64[3] = [0; 3];
    for k in C.L * 7 + 1 {
        let r, nb = tsz(g, if k == C.L * 7 { 7 } else { k % 7 });
        let f = fm[k] as i64;
        off.push(o[f]);
        o[f] += if f == 0 { r * nb } else { r * nb / 8 };
    }
    let sh = Sh { a: s8, b: s4, c: s6, fm: fm, off: off };
    let hd = C.hd;
    let d = C.d;
    let nk = (g.kb - g.ka) * hd;
    let n0 = C.L * ((g.hb - g.ha) * hd + 2 * nk + 2 * hd);
    var kc: f32[>..] = [];
    kc.resize(C.L * C.ctx * nk, 0);
    var vc: f32[>..] = [];
    vc.resize(C.L * C.ctx * nk, 0);
    var all: f32[>..] = [];
    all.resize(C.nw * C.nb * d, 0);
    var pb: f32[>..] = [];
    pb.resize(C.nw * C.nb * d, 0);
    var pn: i64[>..] = [];
    pn.resize(C.nw, -1);
    var X: f32[>..] = [];
    X.resize(C.nb * d, 0);
    var xn: f32[>..] = [];
    xn.resize(C.nb * d, 0);
    var rd = 0;
    var ex = 0;
    var rt: f32[>..] = [];
    var tr: i64[>..] = [];
    var ta = 0;
    var tf = 0;
    var tl = 0;
    var busy = 0;
    var n = 0;
    loop {
        let j = rx<J<T>>(C.sp);
        guard j.op >= 0 else { break; }
        let t0 = clock_ns();
        let nt = j.nt as i64;
        let m = nt * d;
        let pos = j.pos as i64;
        for i in m { X[i] = j.x[i]; }
        var par: i64[>..] = [];
        var dep: i64[>..] = [];
        var xo = m;
        if j.tr == 1 {
            for t in nt { par.push(j.x[xo + t] as i64); dep.push(j.x[xo + nt + t] as i64); }
            xo += 2 * nt;
        }
        for c in j.nc as i64 {
            let src = j.x[xo + 2 * c] as i64;
            let dst = j.x[xo + 2 * c + 1] as i64;
            for l in C.L {
                let a0 = (l * C.ctx + src) * nk;
                let b0 = (l * C.ctx + dst) * nk;
                for i in nk { kc[b0 + i] = kc[a0 + i]; vc[b0 + i] = vc[a0 + i]; }
            }
        }
        rtab(rt, pos + nt);
        for l in C.L {
            let wa = n0 + l * 2 * d;
            for t in nt { rms(X[t * d..(t + 1) * d], nv[wa..wa + d], xn[t * d..(t + 1) * d]); }
            let a0 = clock_ns();
            if tr.len < 60000 { tr.push(a0); }
            var pa = attb<T>(w, g, sh, nv, kc, vc, l, pos, nt, xn[..m], par, dep, rt);
            let e0 = clock_ns();
            ta += e0 - a0;
            if tr.len < 60000 { tr.push(e0); }
            ar<T>(w, pa, rd, X[..m], all, pb, pn, tr, sh, l * 7 + 4);
            ex += clock_ns() - e0;
            rd++;
            for t in nt { rms(X[t * d..(t + 1) * d], nv[wa + d..wa + 2 * d], xn[t * d..(t + 1) * d]); }
            let f0 = clock_ns();
            if tr.len < 60000 { tr.push(f0); }
            var pf = ffnb(g, sh, l, nt, xn[..m]);
            let e1 = clock_ns();
            tf += e1 - f0;
            if tr.len < 60000 { tr.push(e1); }
            ar<T>(w, pf, rd, X[..m], all, pb, pn, tr, sh, if l + 1 < C.L { (l + 1) * 7 } else { C.L * 7 });
            ex += clock_ns() - e1;
            rd++;
        }
        let wo = n0 + C.L * 2 * d;
        let a = j.l as i64;
        for t in a..nt { rms(X[t * d..(t + 1) * d], nv[wo..wo + d], xn[t * d..(t + 1) * d]); }
        let l0 = clock_ns();
        if j.op == 1 { qput(Res { w: w as i32, y: topks(lmb(g, sh, nt - a, xn[a * d..m]), g.va, j.k, nt - a) }); }
        else if j.op == 2 { qput(Res { w: w as i32, y: lmb(g, sh, nt - a, xn[a * d..m]) }); }
        tl += clock_ns() - l0;
        busy += clock_ns() - t0;
        n++;
    }
    qput(St { w: w as i32, busy: busy, n: n, ex: ex, ta: ta, tf: tf, tl: tl, tr: copy(tr) });
}
thread_fn worker(w: i64, s8: QU[], s4: K4[], s6: K6[], fm: i32[], nv: f32[]) {
    const for <K> in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] { if w + 1 == K { wloop<u8[K]>(w, s8, s4, s6, fm, nv); } }
}

var NB = 256;
var LB = 64;
var SP = 2000;
var FB = true;
var PIN = true;
var PFB = 131072;
var emb: Q8[>..];
var LG: f32[>..<];
var CD: f32[>..<];
var vb: u8[>..];
var vo: i64[>..];
var vt: i32[>..];
var mb: u8[>..];
var mo: i64[>..];
var sp: i64[>..];
var t2i: dictionary<const u8[:], i32>;
var mrk: dictionary<const u8[:], i32>;
var u2b: i32[>..];
var b2u: i32[256] = [0; 256];

fn ssz(ty: i64) -> i64 {
    if ty < 2 || ty == 7 { return 1; }
    if ty < 4 { return 2; }
    if ty < 7 { return 4; }
    return 8;
}
fn vsz(f: const u8[:], ty: i64, p: i64) -> i64 {
    if ty == 8 { return 8 + binary::u64le(f, p) as i64; }
    if ty == 9 {
        let et = binary::u32le(f, p) as i64;
        let n = binary::u64le(f, p + 4) as i64;
        var q = p + 12;
        if et == 8 { for i in n { q += 8 + binary::u64le(f, q) as i64; } } else { q += n * ssz(et); }
        return q - p;
    }
    return ssz(ty);
}
fn kfind(f: const u8[:], kvs: const KV[:], key: const u8[:]) -> i64 {
    for k, i in kvs { if f[k.ko..k.ko + k.kl] == key { return i; } }
    return -1;
}
fn kint(f: const u8[:], kvs: const KV[:], key: const u8[:], df: i64) -> i64 {
    let i = kfind(f, kvs, key);
    guard i >= 0 else { return df; }
    let k = kvs[i];
    if k.ty == 0 || k.ty == 7 { return f[k.vo]; }
    if k.ty == 4 { return binary::u32le(f, k.vo); }
    if k.ty == 5 { return binary::u32le(f, k.vo) as! i32; }
    return binary::u64le(f, k.vo) as! i64;
}
fn kflt(f: const u8[:], kvs: const KV[:], key: const u8[:], df: f32) -> f32 {
    let i = kfind(f, kvs, key);
    guard i >= 0 else { return df; }
    let k = kvs[i];
    if k.ty == 6 { return binary::f32_of_bits(binary::u32le(f, k.vo)); }
    return binary::f64_of_bits(binary::u64le(f, k.vo)) as f32;
}
fn kstr(f: const u8[:], kvs: const KV[:], key: const u8[:]) -> const u8[:] {
    let i = kfind(f, kvs, key);
    guard i >= 0 else { return f[0..0]; }
    let p = kvs[i].vo;
    return f[p + 8..p + 8 + binary::u64le(f, p) as i64];
}
fn parse(f: const u8[:], kvs: KV[>..]&, ts: TI[>..]&) -> i64 {
    guard binary::u32le(f, 0) == 1179993927 else { abort("not gguf"); }
    let nt = binary::u64le(f, 8) as i64;
    let nk = binary::u64le(f, 16) as i64;
    var p = 24;
    for i in nk {
        let kl = binary::u64le(f, p) as i64;
        let ty = binary::u32le(f, p + 8 + kl);
        kvs.push(KV { p + 8, kl, ty, p + 12 + kl });
        p += 12 + kl + vsz(f, ty, p + 12 + kl);
    }
    for i in nt {
        let nl = binary::u64le(f, p) as i64;
        let nd = binary::u32le(f, p + 8 + nl);
        let q = p + 12 + nl;
        let e1 = if nd > 1 { binary::u64le(f, q + 8) as i64 } else { 1 };
        ts.push(TI { p + 8, nl, binary::u32le(f, q + nd * 8), binary::u64le(f, q) as i64, e1, binary::u64le(f, q + nd * 8 + 4) as i64 });
        p = q + nd * 8 + 12;
    }
    let al = kint(f, kvs, "general.alignment", 32);
    return (p + al - 1) / al * al;
}
fn tfind(f: const u8[:], ts: const TI[:], name: const u8[:]) -> i64 {
    for t, i in ts { if f[t.no..t.no + t.nl] == name { return i; } }
    return -1;
}

fn rbytes(ty: i64, n: i64) -> i64 {
    if ty == 0 { return n * 4; }
    if ty == 1 || ty == 30 { return n * 2; }
    if ty == 8 { return n / 32 * 34; }
    if ty == 2 { return n / 32 * 18; }
    if ty == 12 { return n / 256 * 144; }
    if ty == 14 { return n / 256 * 210; }
    abort(str("unsupported ggml type ", ty));
}
fn smk(f: const u8[:], o: i64, j: i64) -> i32, i32 {
    if j < 4 { return f[o + j] & 63, f[o + j + 4] & 63; }
    return (f[o + j + 4] & 15) | ((f[o + j - 4] >> 6) << 4), (f[o + j + 4] >> 4) | ((f[o + j] >> 6) << 4);
}
fn deq(f: const u8[:], ty: i64, p: i64, n: i64, y: f32[>..]&) {
    if ty == 0 { for i in n { y.push(binary::f32_of_bits(binary::u32le(f, p + i * 4))); } }
    else if ty == 1 { for i in n { y.push(h2f(binary::u16le(f, p + i * 2))); } }
    else if ty == 30 { for i in n { y.push(binary::f32_of_bits(binary::u16le(f, p + i * 2) as u32 << 16)); } }
    else if ty == 8 {
        for b in n / 32 {
            let o = p + b * 34;
            let d = h2f(binary::u16le(f, o));
            for j in 32 { y.push(d * (f[o + 2 + j] as! i8)); }
        }
    } else if ty == 2 {
        for b in n / 32 {
            let o = p + b * 18;
            let d = h2f(binary::u16le(f, o));
            for j in 16 { y.push(d * ((f[o + 2 + j] & 15) as i32 - 8)); }
            for j in 16 { y.push(d * ((f[o + 2 + j] >> 4) as i32 - 8)); }
        }
    } else if ty == 12 {
        for b in n / 256 {
            let o = p + b * 144;
            let d = h2f(binary::u16le(f, o));
            let dm = h2f(binary::u16le(f, o + 2));
            var q = o + 16;
            for g in 4 {
                let s1, m1 = smk(f, o + 4, g * 2);
                let s2, m2 = smk(f, o + 4, g * 2 + 1);
                let d1 = d * s1;
                let n1 = dm * m1;
                let d2 = d * s2;
                let n2 = dm * m2;
                for l in 32 { y.push(d1 * (f[q + l] & 15) - n1); }
                for l in 32 { y.push(d2 * (f[q + l] >> 4) - n2); }
                q += 32;
            }
        }
    } else if ty == 14 {
        for b in n / 256 {
            let o = p + b * 210;
            let d = h2f(binary::u16le(f, o + 208));
            for h in 2 {
                let ql = o + h * 64;
                let qh = o + 128 + h * 32;
                let sc = o + 192 + h * 8;
                var t: f32[128] = [0.0; 128];
                for l in 32 {
                    let is = l / 16;
                    let a = f[qh + l];
                    t[l] = d * (f[sc + is] as! i8) * (((f[ql + l] & 15) | ((a & 3) << 4)) as i32 - 32);
                    t[l + 32] = d * (f[sc + is + 2] as! i8) * (((f[ql + l + 32] & 15) | (((a >> 2) & 3) << 4)) as i32 - 32);
                    t[l + 64] = d * (f[sc + is + 4] as! i8) * (((f[ql + l] >> 4) | (((a >> 4) & 3) << 4)) as i32 - 32);
                    t[l + 96] = d * (f[sc + is + 6] as! i8) * (((f[ql + l + 32] >> 4) | (((a >> 6) & 3) << 4)) as i32 - 32);
                }
                y.append(t);
            }
        }
    } else { abort(str("unsupported ggml type ", ty)); }
}
fn wcols(f: const u8[:], ty: i64, p: i64, n: i64, c0: i64, c1: i64, o: Q8[>..]&) {
    if ty == 8 {
        for b in c0 / 32..c1 / 32 {
            let s = p + b * 34;
            let r .= o.push(Q8 { d: binary::u16le(f, s), q: [0; 32] });
            for j in 32 { r.q[j] = f[s + 2 + j] as! i8; }
        }
        return;
    }
    var t: f32[>..] = [];
    deq(f, ty, p, n, t);
    qrow(t[c0..c1], o);
}
fn tfmt(ty: i64, t: i64, g: Geo) -> i64 {
    if ty != 12 && ty != 14 { return 0; }
    if t == 3 && ((g.ha * C.hd) % 256 != 0 || (g.hb * C.hd) % 256 != 0) { return 0; }
    if t == 6 && (g.fa % 8 != 0 || g.fb % 8 != 0) { return 0; }
    if t != 3 && t != 6 && C.d % 256 != 0 { return 0; }
    return if ty == 12 { 1 } else { 2 };
}
fn shard(f: const u8[:], ts: const TI[:], tk: const i64[:], base: i64, w: i64, o8: QU[>..]&, o4: K4[>..]&, o6: K6[>..]&, fm: i32[>..]&) {
    let g = geo(w);
    let hd = C.hd;
    for k in C.L * 7 + 1 {
        let t = if k == C.L * 7 { 7 } else { k % 7 };
        let ti = ts[tk[k]];
        let R, N = tdims(k);
        guard ti.ne0 == N && ti.ne1 == R else { abort(str("shape mismatch ", f[ti.no..ti.no + ti.nl])); }
        let rb = rbytes(ti.ty, N);
        let p0 = base + ti.off;
        let fmt = tfmt(ti.ty, t, g);
        fm.push(fmt as i32);
        var r0 = 0;
        var r1 = R;
        var c0 = 0;
        var c1 = N;
        if t == 3 { c0 = g.ha * hd; c1 = g.hb * hd; }
        else if t == 6 { c0 = g.fa * 32; c1 = g.fb * 32; }
        else if t == 0 { r0 = g.ha * hd; r1 = g.hb * hd; }
        else if t < 3 { r0 = 0; r1 = 0; }
        else if t < 6 { r0 = g.fa * 32; r1 = g.fb * 32; }
        else { r0 = g.va; r1 = g.vb; }
        if c1 <= c0 { continue; }
        var rr: i64[>..] = [r0, r1];
        if t == 1 || t == 2 { rr.clear(); for h in g.ka..g.kb { let s0, s1 = ksl(h, w); rr.push(h * hd + s0); rr.push(h * hd + s1); } }
        for ri in rr.len / 2 { for r in rr[2 * ri]..rr[2 * ri + 1] {
            let rp = p0 + r * rb;
            if fmt == 0 {
                var t8: Q8[>..] = [];
                wcols(f, ti.ty, rp, N, c0, c1, t8);
                for e in t8 { let u .= o8.push(QU { d: e.d, q: [0; 32] }); for j in 32 { u.q[j] = (e.q[j] as i32 + 128) as u8; } }
            }
            else if fmt == 2 {
                for b in c0 / 256..c1 / 256 {
                    let q = rp + b * 210;
                    let e .= o6.push(K6 { d: binary::u16le(f, q + 208), sc: [0; 16], ql: [0; 128], qh: [0; 64] });
                    for j in 16 { e.sc[j] = f[q + 192 + j] as! i8; }
                    var v: u8[256] = [0; 256];
                    for h in 2 {
                        for l in 32 {
                            let a = f[q + 128 + h * 32 + l];
                            let p0 = f[q + h * 64 + l];
                            let p1 = f[q + h * 64 + 32 + l];
                            v[h * 128 + l] = (p0 & 15) | ((a & 3) << 4);
                            v[h * 128 + 32 + l] = (p1 & 15) | (((a >> 2) & 3) << 4);
                            v[h * 128 + 64 + l] = (p0 >> 4) | (((a >> 4) & 3) << 4);
                            v[h * 128 + 96 + l] = (p1 >> 4) | (((a >> 6) & 3) << 4);
                        }
                    }
                    for h in 2 {
                        for p in 32 {
                            var w: u8[4] = [0; 4];
                            for k in 4 { let g = 4 * h + k; w[k] = v[16 * ((p >> 2) + 8 * (g & 1)) + 4 * (g >> 1) + (p & 3)]; }
                            e.ql[h * 64 + p] = (w[0] & 15) | ((w[2] & 15) << 4);
                            e.ql[h * 64 + 32 + p] = (w[1] & 15) | ((w[3] & 15) << 4);
                            e.qh[h * 32 + p] = (w[0] >> 4) | ((w[1] >> 4) << 2) | ((w[2] >> 4) << 4) | ((w[3] >> 4) << 6);
                        }
                    }
                }
            }
            else {
                for b in c0 / 256..c1 / 256 {
                    let q = rp + b * 144;
                    let e .= o4.push(K4 { d: binary::u16le(f, q), dm: binary::u16le(f, q + 2), sc: [0; 8], m: [0; 8], q: [0; 128] });
                    for j in 8 { let c, m = smk(f, q + 4, j); e.sc[j] = c as u8; e.m[j] = m as u8; }
                    var v: u8[256] = [0; 256];
                    for g in 4 { for l in 32 { let z = f[q + 16 + g * 32 + l]; v[64 * g + l] = z & 15; v[64 * g + 32 + l] = z >> 4; } }
                    for c in 4 { for p in 32 { let n = 16 * (p >> 2) + 4 * c + (p & 3); e.q[c * 32 + p] = v[n] | (v[n + 128] << 4); } }
                }
            }
        } }
    }
}
fn vsl(f: const u8[:], ts: const TI[:], base: i64, name: const u8[:], n: i64, a: i64, b: i64, o: f32[>..]&) {
    let i = tfind(f, ts, name);
    if i < 0 { for j in b - a { o.push(0); } return; }
    var t: f32[>..] = [];
    deq(f, ts[i].ty, base + ts[i].off, n, t);
    o.append(t[a..b]);
}
fn nvec(f: const u8[:], ts: const TI[:], base: i64, w: i64) -> f32[>..] {
    let g = geo(w);
    let hd = C.hd;
    var o: f32[>..] = [];
    for l in C.L {
        vsl(f, ts, base, str("blk.", l, ".attn_q.bias"), C.nh * hd, g.ha * hd, g.hb * hd, o);
        vsl(f, ts, base, str("blk.", l, ".attn_k.bias"), C.nkv * hd, g.ka * hd, g.kb * hd, o);
        vsl(f, ts, base, str("blk.", l, ".attn_v.bias"), C.nkv * hd, g.ka * hd, g.kb * hd, o);
        vsl(f, ts, base, str("blk.", l, ".attn_q_norm.weight"), hd, 0, hd, o);
        vsl(f, ts, base, str("blk.", l, ".attn_k_norm.weight"), hd, 0, hd, o);
    }
    for l in C.L {
        vsl(f, ts, base, str("blk.", l, ".attn_norm.weight"), C.d, 0, C.d, o);
        vsl(f, ts, base, str("blk.", l, ".ffn_norm.weight"), C.d, 0, C.d, o);
    }
    vsl(f, ts, base, "output_norm.weight", C.d, 0, C.d, o);
    return o;
}

fn u8enc(o: u8[>..]&, c: i32) {
    if c < 128 { o.push(c as u8); }
    else if c < 2048 { o.push((192 | (c >> 6)) as u8); o.push((128 | (c & 63)) as u8); }
    else if c < 65536 { o.push((224 | (c >> 12)) as u8); o.push((128 | ((c >> 6) & 63)) as u8); o.push((128 | (c & 63)) as u8); }
    else { o.push((240 | (c >> 18)) as u8); o.push((128 | ((c >> 12) & 63)) as u8); o.push((128 | ((c >> 6) & 63)) as u8); o.push((128 | (c & 63)) as u8); }
}
fn u8dec(s: const u8[:], i: i64) -> i32, i64 {
    let b = s[i] as i32;
    if b < 192 { return b, 1; }
    if b < 224 && i + 1 < s.len { return ((b & 31) << 6) | (s[i + 1] & 63), 2; }
    if b < 240 && i + 2 < s.len { return ((b & 15) << 12) | ((s[i + 1] & 63) as i32 << 6) | (s[i + 2] & 63), 3; }
    if i + 3 < s.len { return ((b & 7) << 18) | ((s[i + 1] & 63) as i32 << 12) | ((s[i + 2] & 63) as i32 << 6) | (s[i + 3] & 63), 4; }
    return b, 1;
}
fn isn(c: i32) -> bool { (c >= 48 && c <= 57) || c == 178 || c == 179 || c == 185 || (c >= 188 && c <= 190) || (c >= 0x660 && c <= 0x669) || (c >= 0xFF10 && c <= 0xFF19) }
fn iss(c: i32) -> bool { c == 32 || (c >= 9 && c <= 13) || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000 }
fn isl(c: i32) -> bool {
    if c < 128 { return (c >= 65 && c <= 90) || (c >= 97 && c <= 122); }
    if c < 192 { return c == 170 || c == 181 || c == 186; }
    if c == 215 || c == 247 || isn(c) || iss(c) { return false; }
    if (c >= 0x2000 && c <= 0x2BFF) || (c >= 0x3000 && c <= 0x303F) || (c >= 0xFE30 && c <= 0xFE4F) || (c >= 0xFF00 && c <= 0xFF20) || (c >= 0xFF3B && c <= 0xFF40) || (c >= 0xFF5B && c <= 0xFF65) || c >= 0x1F000 { return false; }
    return true;
}
fn pmatch(cp: const i32[:], i: i64) -> i64 {
    let n = cp.len;
    let c = cp[i];
    if c == 39 && i + 1 < n {
        let a = cp[i + 1] | 32;
        if a == 115 || a == 116 || a == 109 || a == 100 { return i + 2; }
        if i + 2 < n {
            let b = cp[i + 2] | 32;
            if (a == 114 && b == 101) || (a == 118 && b == 101) || (a == 108 && b == 108) { return i + 3; }
        }
    }
    var j = i;
    if !isl(c) && !isn(c) && c != 10 && c != 13 { j++; }
    if j < n && isl(cp[j]) { while j < n && isl(cp[j]) { j++; } return j; }
    if isn(c) { return i + 1; }
    j = if c == 32 { i + 1 } else { i };
    var k = j;
    while k < n && !iss(cp[k]) && !isl(cp[k]) && !isn(cp[k]) { k++; }
    if k > j { while k < n && (cp[k] == 10 || cp[k] == 13) { k++; } return k; }
    if iss(c) {
        var e = i;
        while e < n && iss(cp[e]) { e++; }
        var ln = -1;
        for q in i..e { if cp[q] == 10 || cp[q] == 13 { ln = q; } }
        if ln >= 0 { return ln + 1; }
        if e == n || e - i == 1 { return e; }
        return e - 1;
    }
    return i + 1;
}
fn bpe(m: const u8[:], out: i32[>..]&) {
    var bd: i64[>..<] = [];
    var i = 0;
    while i < m.len { bd.push(i); let c, l = u8dec(m, i); i += l; }
    bd.push(m.len);
    var key: u8[>..<] = [];
    loop {
        var best = -1;
        var br: i32 = 2147483647;
        for k in bd.len - 2 {
            key.clear();
            key.append(m[bd[k]..bd[k + 1]]);
            key.push(' ');
            key.append(m[bd[k + 1]..bd[k + 2]]);
            let r = mrk.get(key);
            if r { if r < br { br = r; best = k; } }
        }
        guard best >= 0 else { break; }
        remove_at(bd, best + 1);
    }
    for k in bd.len - 1 {
        let id = t2i.get(m[bd[k]..bd[k + 1]]);
        if id { out.push(id); }
    }
}
fn etext(s: const u8[:], out: i32[>..]&) {
    var cp: i32[>..] = [];
    var bo: i64[>..] = [];
    var i = 0;
    while i < s.len { let c, l = u8dec(s, i); cp.push(c); bo.push(i); i += l; }
    bo.push(s.len);
    var p = 0;
    while p < cp.len {
        let e = pmatch(cp, p);
        var m: u8[>..] = [];
        for b in s[bo[p]..bo[e]] { u8enc(m, b2u[b]); }
        bpe(m, out);
        p = e;
    }
}
fn encode(s: const u8[:], out: i32[>..]&) {
    var i = 0;
    var st = 0;
    while i < s.len {
        var hit = -1;
        for id in sp {
            let t = vb[vo[id]..vo[id + 1]];
            if t.len > 0 && i + t.len <= s.len && s[i..i + t.len] == t { hit = id; break; }
        }
        if hit >= 0 {
            etext(s[st..i], out);
            out.push(hit as i32);
            i += vo[hit + 1] - vo[hit];
            st = i;
        } else { i++; }
    }
    etext(s[st..], out);
}
fn piece(id: i64, o: u8[>..]&) {
    let t = vb[vo[id]..vo[id + 1]];
    if vt[id] == 3 || vt[id] == 4 { o.append(t); return; }
    var i = 0;
    while i < t.len {
        let c, l = u8dec(t, i);
        o.push(if c < 512 { u2b[c] as u8 } else { '?' });
        i += l;
    }
}

fn load(path: const u8[:], nw: i64, ctx: i64, ids: i64[>..]&) {
    var f: u8[>..] = [];
    guard read_file(path, f) else { abort(str("cannot read ", path)); }
    var kvs: KV[>..] = [];
    var ts: TI[>..] = [];
    let base = parse(f, kvs, ts);
    let ar = kstr(f, kvs, "general.architecture");
    let d = kint(f, kvs, str(ar, ".embedding_length"), 0);
    let nh = kint(f, kvs, str(ar, ".attention.head_count"), 0);
    let hd = kint(f, kvs, str(ar, ".attention.key_length"), d / nh);
    let te = tfind(f, ts, "token_embd.weight");
    C = Cfg { L: kint(f, kvs, str(ar, ".block_count"), 0), d: d, ff: kint(f, kvs, str(ar, ".feed_forward_length"), 0),
              nh: nh, nkv: kint(f, kvs, str(ar, ".attention.head_count_kv"), nh), hd: hd, V: ts[te].ne1, nw: nw, ctx: ctx, sp: SP, fg: 1, nb: NB, fb: FB, pin: PIN,
              neox: ar != "llama", qkn: tfind(f, ts, "blk.0.attn_q_norm.weight") >= 0,
              eps: kflt(f, kvs, str(ar, ".attention.layer_norm_rms_epsilon"), 0.00001), base: kflt(f, kvs, str(ar, ".rope.freq_base"), 10000) };
    for i in 65536 { HT[i] = h2f(i as u16); }

    for b in 256 { b2u[b] = if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174 { b as i32 } else { 0 }; }
    var nx: i32 = 256;
    for b in 256 { if b2u[b] == 0 { b2u[b] = nx; nx++; } }
    u2b.resize(512, 0);
    for b in 256 { u2b[b2u[b]] = b as i32; }
    let tki = kfind(f, kvs, "tokenizer.ggml.tokens");
    var p = kvs[tki].vo + 12;
    vo.push(0);
    for i in binary::u64le(f, kvs[tki].vo + 4) as i64 { let l = binary::u64le(f, p) as i64; vb.append(f[p + 8..p + 8 + l]); vo.push(vb.len); p += 8 + l; }
    let tti = kfind(f, kvs, "tokenizer.ggml.token_type");
    for i in vo.len - 1 { vt.push(if tti >= 0 { binary::u32le(f, kvs[tti].vo + 12 + i * 4) as! i32 } else { 1 }); }
    let mi = kfind(f, kvs, "tokenizer.ggml.merges");
    p = kvs[mi].vo + 12;
    mo.push(0);
    for i in binary::u64le(f, kvs[mi].vo + 4) as i64 { let l = binary::u64le(f, p) as i64; mb.append(f[p + 8..p + 8 + l]); mo.push(mb.len); p += 8 + l; }
    for i in vo.len - 1 {
        t2i.insert(vb[vo[i]..vo[i + 1]], i as i32);
        if vt[i] == 3 || vt[i] == 4 { sp.push(i); }
    }
    for i in mo.len - 1 { mrk.insert(mb[mo[i]..mo[i + 1]], i as i32); }

    if GPU { C.nw = 1; gload(f, ts, base, te); return; }
    let rb = rbytes(ts[te].ty, d);
    for r in C.V { wcols(f, ts[te].ty, base + ts[te].off + r * rb, d, 0, d, emb); }

    var kd = false;
    for l in C.L { let i = tfind(f, ts, str("blk.", l, ".ffn_down.weight")); if i >= 0 && (ts[i].ty == 12 || ts[i].ty == 14) { kd = true; } }
    if kd && C.ff % 256 == 0 { C.fg = 8; }
    if C.nw == 0 { C.nw = if C.fg == 8 { 7 } else { 8 }; }
    print(ar, " L=", C.L, " d=", d, " ff=", C.ff, " heads=", nh, "/", C.nkv, " hd=", hd, " vocab=", C.V, " | tensor-parallel x", C.nw);
    var tk: i64[>..] = [];
    let nm = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];
    for l in C.L { for t in 7 { let i = tfind(f, ts, str("blk.", l, ".", nm[t], ".weight")); guard i >= 0 else { abort("missing tensor"); } tk.push(i); } }
    let oi = tfind(f, ts, "output.weight");
    tk.push(if oi >= 0 { oi } else { te });
    for w in C.nw {
        var o8: QU[>..] = [];
        var o4: K4[>..] = [];
        var o6: K6[>..] = [];
        var fm: i32[>..] = [];
        shard(f, ts, tk, base, w, o8, o4, o6, fm);
        if w == 0 { var nk = 0; for v in fm { if v != 0 { nk++; } } print("native k-quant tensors: ", nk, "/", fm.len, " (worker 0)"); }
        ids.push(thread_spawn(worker, w, copy(o8), copy(o4), copy(o6), copy(fm), nvec(f, ts, base, w)));
    }

}

fn bc(op: i64, l: i64, pos: i64, nt: i64, k: i64, x: f32[>..]&, tr: i64, nc: i64) {
    const for <K> in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] { if K <= C.nw { qput(J<u8[K]> { op: op as i32, l: l as i32, pos: pos as i32, nt: nt as i32, k: k as i32, tr: tr as i32, nc: nc as i32, tag: [0; K], x: copy(x) }); } }
}
fn quit(verb: bool) {
    var e: f32[>..] = [];
    bc(-1, 0, 0, 0, 0, e, 0, 0);
    var T: i64[>..] = [];
    var tn = 0;
    for i in C.nw {
        let s = qget<St>();
        if tn == 0 { tn = s.tr.len; T.resize(C.nw * tn, 0); }
        for k in min(tn, s.tr.len) { T[s.w as i64 * tn + k] = s.tr[k]; }
        if verb { print("worker ", s.w, ": ", s.n, " jobs, busy ", s.busy / 1000000, " ms: attn ", s.ta / 1000000, " ffn ", s.tf / 1000000, " lm ", s.tl / 1000000, " allreduce ", s.ex / 1000000, " ms"); }
    }
    guard verb && tn >= 24 * C.L else { return; }
    let nl = tn / 12 - 1;
    var snd: f64 = 0;
    var dly: f64 = 0;
    var prc: f64 = 0;
    var pst: f64 = 0;
    var idle: f64 = 0;
    var cnt = 0;
    var ih: f64[2] = [0.0; 2];
    var ch: i64[2] = [0; 2];
    var slow: i64[32] = [0; 32];
    var dur: f64[64] = [0.0; 64];
    for i in C.L..nl {
        for h in 2 {
            if h == 1 && i % C.L == C.L - 1 { continue; }
            let b = i * 12 + h * 6;
            var sw = 0;
            for w in C.nw { if T[w * tn + b + 2] > T[sw * tn + b + 2] { sw = w; } }
            let sent = T[sw * tn + b + 3];
            snd += sent - T[sw * tn + b + 2];
            var dsum = 0;
            var psum = 0;
            var qsum = 0;
            var isum = 0;
            for w in C.nw {
                dsum += T[w * tn + b + 4] - sent;
                psum += T[w * tn + b + 5] - T[w * tn + b + 4];
                qsum += T[w * tn + b + 6] - T[w * tn + b + 5];
                isum += T[sw * tn + b + 2] - T[w * tn + b + 2];
            }
            dly += dsum / C.nw;
            prc += psum / C.nw;
            pst += qsum / C.nw;
            idle += isum / C.nw;
            ih[h] += isum / C.nw;
            ch[h]++;
            slow[h * 16 + sw]++;
            for w in C.nw { dur[h * 32 + w] += T[w * tn + b + 1] - T[w * tn + b]; }
            cnt++;
        }
    }
    for h in 2 {
        var o: u8[>..] = [];
        format(o, if h == 0 { "attn" } else { "ffn " }, " rounds: idle ", ih[h] / ch[h] / 1000, " us; slowest-count per worker:");
        for w in C.nw { format(o, " ", slow[h * 16 + w]); }
        format(o, "; mean compute us per worker:");
        for w in C.nw { format(o, " ", (dur[h * 32 + w] / ch[h] / 1000) as i64); }
        print(o);
    }
    print("rounds ", cnt, ": slowest worker's sends ", snd / cnt / 1000, " us | delivery to others (last recv - slowest sent) ", dly / cnt / 1000, " us | sum ", prc / cnt / 1000, " us | post (norm etc.) ", pst / cnt / 1000, " us | others idle waiting for slowest ", idle / cnt / 1000, " us");
}
var CK = 1;
var TP: i64[>..<];
var TD: i64[>..<];
var KCP: i64[>..<];
var TB = 8;
var MB = 4;
var DK = 7;
fn fwd(tk: const i32[:], pos: i64, op: i64, a: i64, k: i64) {
    let d = C.d;
    let nb = d / 32;
    let nt = tk.len;
    guard pos + nt <= C.ctx else { abort("context full"); }
    var x: f32[>..] = [];
    x.resize(nt * d, 0);
    for t, i in tk { for b in nb { let e = emb[t * nb + b]; let s = HT[e.d]; for j in 32 { x[i * d + b * 32 + j] = s * e.q[j]; } } }
    var tr = 0;
    if TP.len == nt && nt > 1 {
        for v in TP { x.push(v); }
        for v in TD { x.push(v); }
        tr = 1;
    }
    let nc = KCP.len / 2;
    for v in KCP { x.push(v); }
    KCP.clear();
    bc(op, a, pos, nt, k, x, tr, nc);
    let u = nt - a;
    if op == 1 {
        let m = C.nw * k * 2;
        CD.clear();
        CD.resize(u * m, 0);
        CK = k;
        for i in C.nw {
            let r = rx<Res>(C.sp);
            let o = r.w as i64 * k * 2;
            for v in u { for j in k * 2 { CD[v * m + o + j] = r.y[v * k * 2 + j]; } }
        }
    } else if op == 2 {
        LG.clear();
        LG.resize(u * C.V, 0);
        for i in C.nw {
            let r = rx<Res>(C.sp);
            let g = geo(r.w as i64);
            let n = g.vb - g.va;
            for t in u { for j in n { LG[t * C.V + g.va + j] = r.y[t * n + j]; } }
        }
    }
}
fn better(c: const f32[:], a: i64, b: i64) -> bool { c[a * 2] > c[b * 2] || (c[a * 2] == c[b * 2] && c[a * 2 + 1] < c[b * 2 + 1]) }
fn sample(u: i64, temp: f32, topk: i64, topp: f32, r: rng&) -> i64 {
    let m = C.nw * CK * 2;
    let c = CD[u * m..(u + 1) * m];
    var ix: i64[>..<] = [];
    for i in c.len / 2 {
        var j = ix.len;
        while j > 0 && better(c, i, ix[j - 1]) { j--; }
        if j < topk { insert_at(ix, j, i); if ix.len > topk { ix.pop(); } }
    }
    if temp <= 0 { return c[ix[0] * 2 + 1] as i64; }
    var pr: f64[>..] = [];
    var s: f64 = 0;
    for i in ix { let e = exp((c[i * 2] - c[ix[0] * 2]) as f64 / temp); pr.push(e); s += e; }
    var cu: f64 = 0;
    var n = 0;
    while n < pr.len { cu += pr[n] / s; n++; if cu >= topp { break; } }
    var tot: f64 = 0;
    for k in n { tot += pr[k]; }
    var v = rand_flt(r) * tot;
    for k in n { v -= pr[k]; if v <= 0 { return c[ix[k] * 2 + 1] as i64; } }
    return c[ix[n - 1] * 2 + 1] as i64;
}
fn tput(tk: i32[>..]&, cont: const i32[:], lim: i64) {
    var cur = 0;
    for v, i in cont {
        if i >= lim { return; }
        var nx = -1;
        for c in 1..tk.len { if TP[c] == cur && tk[c] == v { nx = c; break; } }
        if nx < 0 {
            if tk.len >= TB { return; }
            tk.push(v);
            TP.push(cur);
            TD.push(TD[cur] + 1);
            nx = tk.len - 1;
        }
        cur = nx;
    }
}
fn dtree(h: const i32[:], km: i64, tk: i32[>..]&) {
    guard km > 0 else { return; }
    let L = h.len;
    var got = 0;
    var nn = min(4, L - 1);
    while nn >= 2 && got < MB {
        let pat = h[L - nn..L];
        var j = L - nn - 1;
        while j >= 0 && got < MB {
            if h[j..j + nn] == pat {
                tput(tk, h[j + nn..min(j + nn + TB, L)], if got == 0 { km } else { TB });
                got++;
            }
            j--;
        }
        nn--;
    }
}
fn draft(h: const i32[:], km: i64, o: i32[>..]&) {
    guard km > 0 else { return; }
    let L = h.len;
    var nn = min(4, L - 1);
    while nn >= 2 {
        let pat = h[L - nn..L];
        var j = L - nn - 1;
        while j >= 0 {
            if h[j..j + nn] == pat { o.append(h[j + nn..min(j + nn + km, L)]); return; }
            j--;
        }
        nn--;
    }
}
fn emit(t: i64, o: u8[>..]&) {
    o.clear();
    piece(t, o);
    write_stdout(o);
    flush_stdout();
}
fn prefill(tk: const i32[:], pos: i64, k: i64) {
    var b0 = 0;
    while b0 < tk.len {
        let b1 = min(b0 + NB, tk.len);
        if b1 == tk.len { fwd(tk[b0..b1], pos + b0, 1, b1 - b0 - 1, k); } else { fwd(tk[b0..b1], pos + b0, 0, 0, 0); }
        b0 = b1;
    }
}

fn ppl(path: const u8[:]) {
    var s: u8[>..] = [];
    guard read_file(path, s) else { abort(str("cannot read ", path)); }
    var tk: i32[>..] = [];
    encode(s, tk);
    let n = C.ctx;
    let nc = tk.len / n;
    let h = n / 2;
    var nll: f64 = 0;
    var cnt = 0;
    let t0 = clock();
    print(tk.len, " tokens, ", nc, " chunks of ", n);
    for c in nc {
        let st = c * n;
        var b0 = 0;
        while b0 < n {
            var b1 = min(b0 + NB, n);
            let a0 = max(b0, h);
            if GPU {
                gset(tk[st + b0], b0);
                gstep(false);
                b1 = b0 + 1;
                if b0 >= h { LG.resize(C.V, 0); cuda::download(GLG, LG); }
            } else if b1 > h { fwd(tk[st + b0..st + b1], b0, 2, a0 - b0, 0); } else { fwd(tk[st + b0..st + b1], b0, 0, 0, 0); }
            for t in a0..min(b1, n - 1) {
                let lg = LG[(t - a0) * C.V..(t - a0 + 1) * C.V];
                var mx = lg[0];
                for v in lg { mx = max(mx, v); }
                var se: f64 = 0;
                for v in lg { se += exp(v - mx); }
                nll -= lg[tk[st + t + 1]] - mx - log(se);
                cnt++;
            }
            b0 = b1;
        }
        var o: u8[>..] = [];
        format(o, "[", c + 1, "]");
        format_flt(o, exp(nll / cnt), 4);
        format(o, " ");
        write_stdout(o);
        flush_stdout();
    }
    var o: u8[>..] = [];
    format(o, "\nPPL = ");
    format_flt(o, exp(nll / cnt), 4);
    format(o, " over ", cnt, " tokens, ");
    format_flt(o, nc * n / (clock() - t0), 1);
    format(o, " tok/s\n");
    write_stdout(o);
}

// f: 0 Q8_0 (q 8 words, s an f16 a 32-block), 1 Q4_K (raw 36-word blocks),
// 2 Q6_K (53-word blocks: ql, qh, scales, d).
struct GM { f: i64, q: cuda::buf<i32>, s: cuda::buf<u16>, u: cuda::buf<i32>, t: cuda::buf<f32>, n: cuda::buf<f32>, rows: i64, k: i64 }
var GPU = false;
var GW: GM[>..];
var GVO: i64[>..];
var GV: cuda::buf<f32>;
var GX: cuda::buf<f32>;
var GQKV: cuda::buf<f32>;
var GPM: cuda::buf<f32>;
var GPA: cuda::buf<f32>;
var GLG: cuda::buf<f32>;
var GKC: cuda::buf<f32>;
var GVC: cuda::buf<f32>;
// For kfa, in halves: K as GKC, V by layer, dimension, then position.
var GKH: cuda::buf<u16>;
var GVT: cuda::buf<u16>;
var GRT: cuda::buf<f32>;
var GXQ: cuda::buf<i32>;
var GXS: cuda::buf<f32>;
var GXM: cuda::buf<f32>;
var GHQ: cuda::buf<i32>;
var GHS: cuda::buf<f32>;
var GHM: cuda::buf<f32>;
var GO: cuda::buf<i32>;
var GCNT: cuda::buf<i32>;
var GBT: cuda::buf<i32>;
var GBS: cuda::buf<i32>;
var GBO: cuda::buf<i32>;
var GBX: cuda::buf<f32>;
var GBQKV: cuda::buf<f32>;
var GBQ: cuda::buf<i32>;
var GBQS: cuda::buf<f32>;
var GBGU: cuda::buf<f32>;
var GBH: cuda::buf<i32>;
var GBHS: cuda::buf<f32>;
var GBPM: cuda::buf<f32>;
var GBPA: cuda::buf<f32>;
var GBLG: cuda::buf<f32>;
var GBCNT: cuda::buf<i32>;
var GBSK: cuda::buf<f32>;
var GBP: u64[5] = [0; 5];
var GBL: u64 = 0;
var GG: u64 = 0;
var GA: u64 = 0;
var GP = false;
var GT: f64[16] = [0.0; 16];
var GC: f64 = 0;
fn gt(k: i64) { if GP { cuda::sync(); let t = clock(); GT[k] += t - GC; GC = t; } }

// v quantized across the warp, a lane a value: block b's 8 words, its scale
// and its scale times the sum of its quants.
fn wq32(v: f32, l: i64, b: i64, o: i32[:], os: f32[:], om: f32[:]) {
    var m = abs(v);
    m = max(m, cuda_shfl_xor(m, 16));
    m = max(m, cuda_shfl_xor(m, 8));
    m = max(m, cuda_shfl_xor(m, 4));
    m = max(m, cuda_shfl_xor(m, 2));
    m = max(m, cuda_shfl_xor(m, 1));
    let id: f32 = if m != 0 { 127 / m } else { 0 };
    let q = round_even(v * id) as! i32;
    var w = (q & 255) << ((l % 4) * 8);
    w = w | cuda_shfl_xor(w, 1);
    w = w | cuda_shfl_xor(w, 2);
    let qs = wsum(q as f32);
    if l % 4 == 0 { o[b * 8 + l / 4] = w; }
    if l == 0 { os[b] = m / 127; om[b] = m / 127 * qs; }
}
// A batch's scales: token t's block b at bsi(t, b), its scale times the sum
// of its quants 1024 on: a pair of blocks' scales for 64 tokens are 512 bytes
// in a row.
fn bsi(t: i64, b: i64) -> i64 { return (b / 2) * 2048 + t * 2 + b % 2; }
// wq32 for a batch: token t's block b, rows kb blocks long, scales as bsi lays
// them out for a big batch, else token by token, each min-sum after its scale.
fn wq32t(v: f32, l: i64, t: i64, b: i64, kb: i64, big: bool, o: i32[:], os: f32[:]) {
    var m = abs(v);
    m = max(m, cuda_shfl_xor(m, 16));
    m = max(m, cuda_shfl_xor(m, 8));
    m = max(m, cuda_shfl_xor(m, 4));
    m = max(m, cuda_shfl_xor(m, 2));
    m = max(m, cuda_shfl_xor(m, 1));
    let id: f32 = if m != 0 { 127 / m } else { 0 };
    let q = round_even(v * id) as! i32;
    var w = (q & 255) << ((l % 4) * 8);
    w = w | cuda_shfl_xor(w, 1);
    w = w | cuda_shfl_xor(w, 2);
    let qs = wsum(q as f32);
    if l % 4 == 0 { o[(t * kb + b) * 8 + l / 4] = w; }
    if l == 0 { let i = if big { bsi(t, b) } else { (t * kb + b) * 2 }; os[i] = m / 127; os[i + (if big { 1024 } else { 1 })] = m / 127 * qs; }
}
fn wsum(v: f32) -> f32 {
    var s = v + cuda_shfl_xor(v, 16);
    s += cuda_shfl_xor(s, 8);
    s += cuda_shfl_xor(s, 4);
    s += cuda_shfl_xor(s, 2);
    return s + cuda_shfl_xor(s, 1);
}
fn kb(a: i32, b: i32, c: i32, i: i64) -> i32 {
    let w = if i < 4 { a } else if i < 8 { b } else { c };
    return (w >> ((i % 4) * 8)) & 255;
}
// Q4_K's 6-bit scale and min of sub-block j, from its 12 scale bytes.
fn ksm(a: i32, b: i32, c: i32, j: i64) -> i32, i32 {
    if j < 4 { return kb(a, b, c, j) & 63, kb(a, b, c, j + 4) & 63; }
    return (kb(a, b, c, j + 4) & 15) | ((kb(a, b, c, j - 4) >> 6) << 4), (kb(a, b, c, j + 4) >> 4) | ((kb(a, b, c, j) >> 6) << 4);
}
// Q4_K unit u (0..7) of the block at wq[bo..]: 16 values of sub-blocks 2p
// and 2p + 1 (p = u / 2), against block blk of the activations.
fn d4k(wq: const i32[:], bo: i64, u: i64, xq: const i32[:], xs: const f32[:], xm: const f32[:], blk: i64) -> f32 {
    let p = u / 2;
    let hf = u % 2;
    let hv = cuda_ld4(wq, bo);
    let hd = hv[0];
    let s0, n0 = ksm(hv[1], hv[2], hv[3], 2 * p);
    let s1, n1 = ksm(hv[1], hv[2], hv[3], 2 * p + 1);
    let wv = cuda_ld4(wq, bo + 4 + p * 8 + hf * 4);
    let xb = blk * 64 + p * 16 + hf * 4;
    let xa = cuda_ld4(xq, xb);
    let xc = cuda_ld4(xq, xb + 8);
    var lo: i32 = 0;
    var hi: i32 = 0;
    for i in 4 {
        lo = cuda_dp4a(wv[i] & 0x0F0F0F0F, xa[i], lo);
        hi = cuda_dp4a((wv[i] >> 4) & 0x0F0F0F0F, xc[i], hi);
    }
    let b0 = blk * 8 + 2 * p;
    var r = cuda_h2f((hd & 65535) as! u16) * (s0 * lo * xs[b0] + s1 * hi * xs[b0 + 1]);
    if hf == 0 { r -= cuda_h2f(((hd >> 16) & 65535) as! u16) * (n0 * xm[b0] + n1 * xm[b0 + 1]); }
    return r;
}
// Q6_K unit u (0..15): positions 4 (u % 8).. of the four 32-groups of half
// u / 8.
fn d6k(wq: const i32[:], bo: i64, u: i64, xq: const i32[:], xs: const f32[:], blk: i64) -> f32 {
    let n = u / 8;
    let l4 = u % 8;
    let qa = wq[bo + 16 * n + l4];
    let qb = wq[bo + 16 * n + 8 + l4];
    let qh = wq[bo + 32 + 8 * n + l4];
    let xb = blk * 64 + n * 32 + l4;
    let b0 = blk * 8 + n * 4;
    let si = 8 * n + l4 / 4;
    var r: f32 = 0;
    for g in 4 {
        let v = if g == 0 { (qa & 0x0F0F0F0F) | ((qh & 0x03030303) << 4) }
                else if g == 1 { (qb & 0x0F0F0F0F) | (((qh >> 2) & 0x03030303) << 4) }
                else if g == 2 { ((qa >> 4) & 0x0F0F0F0F) | (((qh >> 4) & 0x03030303) << 4) }
                else { ((qb >> 4) & 0x0F0F0F0F) | (((qh >> 6) & 0x03030303) << 4) };
        let x = xq[xb + g * 8];
        let s = si + 2 * g;
        let sc = (wq[bo + 48 + s / 4] << (24 - (s % 4) * 8)) >> 24;
        r += (sc * (cuda_dp4a(v, x, 0) - cuda_dp4a(0x20202020, x, 0))) * xs[b0 + g];
    }
    return cuda_h2f((wq[bo + 52] & 65535) as! u16) * r;
}
// Lane l's part of row r times the activations.
inline fn rdot(f: i64, wq: const i32[:], ws: const u16[:], r: i64, k: i64, xq: const i32[:], xs: const f32[:], xm: const f32[:], l: i64) -> f32 {
    var acc: f32 = 0;
    if f == 0 {
        let k4 = k / 4;
        let rb = r * k4;
        let rs = r * (k / 32);
        var j = l * 4;
        while j < k4 {
            let a = cuda_ld4(wq, rb + j);
            let x = cuda_ld4(xq, j);
            let iv = cuda_dp4a(a[3], x[3], cuda_dp4a(a[2], x[2], cuda_dp4a(a[1], x[1], cuda_dp4a(a[0], x[0], 0))));
            acc += iv * cuda_h2f(ws[rs + j / 8]) * xs[j / 8];
            j += 128;
        }
        return acc;
    }
    let nb = k / 256;
    var u = l;
    if f == 1 {
        while u < nb * 8 { acc += d4k(wq, (r * nb + u / 8) * 36, u % 8, xq, xs, xm, u / 8); u += 32; }
        return acc;
    }
    while u < nb * 16 { acc += d6k(wq, (r * nb + u / 16) * 53, u % 16, xq, xs, u / 16); u += 32; }
    return acc;
}
// Rows ra and rb together, their loads in flight at once.
inline fn rdot2(f: i64, wq: const i32[:], ws: const u16[:], ra: i64, rb: i64, k: i64, xq: const i32[:], xs: const f32[:], xm: const f32[:], l: i64) -> f32, f32 {
    var a: f32 = 0;
    var b: f32 = 0;
    if f == 0 {
        let k4 = k / 4;
        let nb = k / 32;
        var j = l * 4;
        while j < k4 {
            let g = cuda_ld4(wq, ra * k4 + j);
            let u = cuda_ld4(wq, rb * k4 + j);
            let x = cuda_ld4(xq, j);
            let sx = xs[j / 8];
            a += cuda_dp4a(g[3], x[3], cuda_dp4a(g[2], x[2], cuda_dp4a(g[1], x[1], cuda_dp4a(g[0], x[0], 0)))) * cuda_h2f(ws[ra * nb + j / 8]) * sx;
            b += cuda_dp4a(u[3], x[3], cuda_dp4a(u[2], x[2], cuda_dp4a(u[1], x[1], cuda_dp4a(u[0], x[0], 0)))) * cuda_h2f(ws[rb * nb + j / 8]) * sx;
            j += 128;
        }
        return a, b;
    }
    let nb = k / 256;
    var u = l;
    if f == 1 {
        while u < nb * 8 {
            a += d4k(wq, (ra * nb + u / 8) * 36, u % 8, xq, xs, xm, u / 8);
            b += d4k(wq, (rb * nb + u / 8) * 36, u % 8, xq, xs, xm, u / 8);
            u += 32;
        }
        return a, b;
    }
    while u < nb * 16 {
        a += d6k(wq, (ra * nb + u / 16) * 53, u % 16, xq, xs, u / 16);
        b += d6k(wq, (rb * nb + u / 16) * 53, u % 16, xq, xs, u / 16);
        u += 32;
    }
    return a, b;
}
cuda fn kemb(f: i64, q: const i32[:], s: const u16[:], x: f32[:], st: const i32[:], k: i64) {
    let j = cuda_block() * cuda_block_dim() + cuda_thread();
    if j >= k { return; }
    let row: i64 = st[0];
    if f == 0 {
        let w = q[row * (k / 4) + j / 4];
        x[j] = cuda_h2f(s[row * (k / 32) + j / 32]) * ((w << ((3 - j % 4) * 8)) >> 24);
        return;
    }
    let nb = k / 256;
    let w = j % 256;
    if f == 1 {
        let bo = (row * nb + j / 256) * 36;
        let js = w / 32;
        let l = w % 32;
        let sc, mn = ksm(q[bo + 1], q[bo + 2], q[bo + 3], js);
        let v = (q[bo + 4 + (js / 2) * 8 + l / 4] >> ((l % 4) * 8 + (js % 2) * 4)) & 15;
        let hd = q[bo];
        x[j] = cuda_h2f((hd & 65535) as! u16) * (sc * v) - cuda_h2f(((hd >> 16) & 65535) as! u16) * mn;
        return;
    }
    let bo = (row * nb + j / 256) * 53;
    let n = w / 128;
    let g = (w % 128) / 32;
    let l = w % 32;
    let lo = (q[bo + 16 * n + (g % 2) * 8 + l / 4] >> ((l % 4) * 8 + (g / 2) * 4)) & 15;
    let hi = (q[bo + 32 + 8 * n + l / 4] >> ((l % 4) * 8 + 2 * g)) & 3;
    let si = 8 * n + l / 16 + 2 * g;
    let sc = (q[bo + 48 + si / 4] << (24 - (si % 4) * 8)) >> 24;
    x[j] = cuda_h2f((q[bo + 52] & 65535) as! u16) * (sc * ((lo | (hi << 4)) - 32));
}
cuda fn knq(x: const f32[:], w: const f32[:], wo: i64, o: i32[:], os: f32[:], om: f32[:], d: i64, eps: f32) {
    let t = cuda_thread();
    let bd = cuda_block_dim();
    let sh = cuda_shared<f32>(32);
    var s: f32 = 0;
    var i = t;
    while i < d { s += x[i] * x[i]; i += bd; }
    s = wsum(s);
    if t % 32 == 0 { sh[t / 32] = s; }
    cuda_sync();
    var tot: f32 = 0;
    for k in bd / 32 { tot += sh[k]; }
    let sc: f32 = 1 / sqrt(tot / d + eps);
    i = t;
    while i < d { wq32(x[i] * sc * w[wo + i], t % 32, i / 32, o, os, om); i += bd; }
}
// Rows ro.. of o: set (mode 0), added to (1), or set plus bias[bo + row] (2).
inline fn mvb(f: i64, wq: const i32[:], ws: const u16[:], xq: const i32[:], xs: const f32[:], xm: const f32[:], o: f32[:], bias: const f32[:], bo: i64, ro: i64, rows: i64, k: i64, mode: i64) {
    let r = cuda_block() * (cuda_block_dim() / 32) + cuda_thread() / 32;
    if r >= rows { return; }
    let l = cuda_thread() % 32;
    let acc = wsum(rdot(f, wq, ws, r, k, xq, xs, xm, l));
    if l == 0 {
        let y = ro + r;
        if mode == 1 { o[y] += acc; } else if mode == 2 { o[y] = acc + bias[bo + y]; } else { o[y] = acc; }
    }
}
cuda fn kmv0(wq: const i32[:], ws: const u16[:], xq: const i32[:], xs: const f32[:], xm: const f32[:], o: f32[:], bias: const f32[:], bo: i64, ro: i64, rows: i64, k: i64, mode: i64) { mvb(0, wq, ws, xq, xs, xm, o, bias, bo, ro, rows, k, mode); }
cuda fn kmv1(wq: const i32[:], ws: const u16[:], xq: const i32[:], xs: const f32[:], xm: const f32[:], o: f32[:], bias: const f32[:], bo: i64, ro: i64, rows: i64, k: i64, mode: i64) { mvb(1, wq, ws, xq, xs, xm, o, bias, bo, ro, rows, k, mode); }
cuda fn kmv2(wq: const i32[:], ws: const u16[:], xq: const i32[:], xs: const f32[:], xm: const f32[:], o: f32[:], bias: const f32[:], bo: i64, ro: i64, rows: i64, k: i64, mode: i64) { mvb(2, wq, ws, xq, xs, xm, o, bias, bo, ro, rows, k, mode); }
// Gate and up rows r and ff + r, silu(g) * u: a warp a row, quantized 32
// rows a block.
inline fn gub(f: i64, wq: const i32[:], ws: const u16[:], xq: const i32[:], xs: const f32[:], xm: const f32[:], o: i32[:], os: f32[:], om: f32[:], ff: i64, k: i64) {
    let w = cuda_thread() / 32;
    let l = cuda_thread() % 32;
    let hs = cuda_shared<f32>(32);
    // The block's 32 rows over its warps: 16 of them (Q8_0, for registers) take two each.
    let nw = cuda_block_dim() / 32;
    for i in 32 / nw {
        let r = cuda_block() * 32 + w + i * nw;
        var g, u = rdot2(f, wq, ws, r, ff + r, k, xq, xs, xm, l);
        g = wsum(g);
        u = wsum(u);
        if l == 0 { hs[w + i * nw] = g / (1 + exp(-g)) * u; }
    }
    cuda_sync();
    if w == 0 { wq32(hs[l], l, cuda_block(), o, os, om); }
}
cuda fn kgu0(wq: const i32[:], ws: const u16[:], xq: const i32[:], xs: const f32[:], xm: const f32[:], o: i32[:], os: f32[:], om: f32[:], ff: i64, k: i64) { gub(0, wq, ws, xq, xs, xm, o, os, om, ff, k); }
cuda fn kgu1(wq: const i32[:], ws: const u16[:], xq: const i32[:], xs: const f32[:], xm: const f32[:], o: i32[:], os: f32[:], om: f32[:], ff: i64, k: i64) { gub(1, wq, ws, xq, xs, xm, o, os, om, ff, k); }
cuda fn kgu2(wq: const i32[:], ws: const u16[:], xq: const i32[:], xs: const f32[:], xm: const f32[:], o: i32[:], os: f32[:], om: f32[:], ff: i64, k: i64) { gub(2, wq, ws, xq, xs, xm, o, os, om, ff, k); }
cuda fn krope(qkv: f32[:], kc: f32[:], vc: f32[:], kh: u16[:], vt: u16[:], rt: const f32[:], st: const i32[:], lc: i64, nh: i64, nkv: i64, hd: i64, neox: bool, ctx: i64, lv: i64) {
    let i = cuda_block() * cuda_block_dim() + cuda_thread();
    let pos: i64 = st[1];
    let co = (lc + pos) * nkv * hd;
    let h2 = hd / 2;
    let nq = nh * h2;
    let nk = nkv * h2;
    if i >= nq + 2 * nk { return; }
    if i >= nq + nk {
        let p = (i - nq - nk) * 2;
        let src = nh * hd + nkv * hd + p;
        vc[co + p] = qkv[src];
        vc[co + p + 1] = qkv[src + 1];
        vt[lv + p * ctx + pos] = cuda_f2h(qkv[src]);
        vt[lv + (p + 1) * ctx + pos] = cuda_f2h(qkv[src + 1]);
        return;
    }
    let hh = i / h2;
    let e = i % h2;
    let ia = if neox { hh * hd + e } else { hh * hd + 2 * e };
    let ib = if neox { ia + h2 } else { ia + 1 };
    let cs = rt[pos * hd + 2 * e];
    let sn = rt[pos * hd + 2 * e + 1];
    let a = qkv[ia];
    let b = qkv[ib];
    let ra = a * cs - b * sn;
    let rb = a * sn + b * cs;
    qkv[ia] = ra;
    qkv[ib] = rb;
    if hh >= nh {
        kc[co + ia - nh * hd] = ra;
        kc[co + ib - nh * hd] = rb;
        kh[co + ia - nh * hd] = cuda_f2h(ra);
        kh[co + ib - nh * hd] = cuda_f2h(rb);
    }
}
// Flash decoding: block h * 16 + c takes positions c * cs.., four warps
// interleaved, and leaves its softmax max, sum and unscaled output.
cuda fn katt(qkv: const f32[:], kc: const f32[:], vc: const f32[:], pm: f32[:], pa: f32[:], st: const i32[:], lo: i64, kvd: i64, gq: i64, s0: f32, cnt: i32[:], ao: i32[:], aos: f32[:], aom: f32[:]) {
    let n: i64 = st[1] + 1;
    let cs = max(32, (n + 15) / 16);
    let h = cuda_block() / 16;
    let c = cuda_block() % 16;
    if c * cs >= n { return; }
    let t = cuda_thread();
    let w = t / 32;
    let l = t % 32;
    let kb = lo + (h / gq) * 128 + l * 4;
    let qo = h * 128 + l * 4;
    let q0 = qkv[qo] * s0;
    let q1 = qkv[qo + 1] * s0;
    let q2 = qkv[qo + 2] * s0;
    let q3 = qkv[qo + 3] * s0;
    var m: f32 = -1e30;
    var sum: f32 = 0;
    var o0: f32 = 0;
    var o1: f32 = 0;
    var o2: f32 = 0;
    var o3: f32 = 0;
    let a1 = min(c * cs + cs, n);
    var s = c * cs + w;
    // Four positions a step: their dot products reduce side by side and the
    // running max moves once.
    while s + 12 < a1 {
        let ka = kb + s * kvd;
        let kc4 = kb + (s + 4) * kvd;
        let kd = kb + (s + 8) * kvd;
        let ke = kb + (s + 12) * kvd;
        let pa0 = wsum(q0 * kc[ka] + q1 * kc[ka + 1] + q2 * kc[ka + 2] + q3 * kc[ka + 3]);
        let pa1 = wsum(q0 * kc[kc4] + q1 * kc[kc4 + 1] + q2 * kc[kc4 + 2] + q3 * kc[kc4 + 3]);
        let pa2 = wsum(q0 * kc[kd] + q1 * kc[kd + 1] + q2 * kc[kd + 2] + q3 * kc[kd + 3]);
        let pa3 = wsum(q0 * kc[ke] + q1 * kc[ke + 1] + q2 * kc[ke + 2] + q3 * kc[ke + 3]);
        let mn = max(max(m, max(pa0, pa1)), max(pa2, pa3));
        let f = exp(m - mn);
        let e0 = exp(pa0 - mn);
        let e1 = exp(pa1 - mn);
        let e2 = exp(pa2 - mn);
        let e3 = exp(pa3 - mn);
        sum = sum * f + ((e0 + e1) + (e2 + e3));
        o0 = o0 * f + ((e0 * vc[ka] + e1 * vc[kc4]) + (e2 * vc[kd] + e3 * vc[ke]));
        o1 = o1 * f + ((e0 * vc[ka + 1] + e1 * vc[kc4 + 1]) + (e2 * vc[kd + 1] + e3 * vc[ke + 1]));
        o2 = o2 * f + ((e0 * vc[ka + 2] + e1 * vc[kc4 + 2]) + (e2 * vc[kd + 2] + e3 * vc[ke + 2]));
        o3 = o3 * f + ((e0 * vc[ka + 3] + e1 * vc[kc4 + 3]) + (e2 * vc[kd + 3] + e3 * vc[ke + 3]));
        m = mn;
        s += 16;
    }
    while s < a1 {
        let ko = kb + s * kvd;
        let p = wsum(q0 * kc[ko] + q1 * kc[ko + 1] + q2 * kc[ko + 2] + q3 * kc[ko + 3]);
        let mn = max(m, p);
        let f = exp(m - mn);
        let e = exp(p - mn);
        sum = sum * f + e;
        o0 = o0 * f + e * vc[ko];
        o1 = o1 * f + e * vc[ko + 1];
        o2 = o2 * f + e * vc[ko + 2];
        o3 = o3 * f + e * vc[ko + 3];
        m = mn;
        s += 4;
    }
    let shm = cuda_shared<f32>(8);
    let sho = cuda_shared<f32>(512);
    if l == 0 { shm[w] = m; shm[4 + w] = sum; }
    sho[w * 128 + l * 4] = o0;
    sho[w * 128 + l * 4 + 1] = o1;
    sho[w * 128 + l * 4 + 2] = o2;
    sho[w * 128 + l * 4 + 3] = o3;
    cuda_sync();
    let mx = max(max(shm[0], shm[1]), max(shm[2], shm[3]));
    var sl: f32 = 0;
    var sa: f32 = 0;
    for k in 4 { let f = exp(shm[k] - mx); sl += shm[4 + k] * f; sa += sho[k * 128 + t] * f; }
    let pi = cuda_block();
    pa[pi * 128 + t] = sa;
    if t == 0 { pm[pi * 2] = mx; pm[pi * 2 + 1] = sl; }
    // The head's last block to finish merges its partials, a warp a 32 dims.
    cuda_fence();
    let lb = cuda_shared<i32>(1);
    let ns = (n + cs - 1) / cs;
    if t == 0 { lb[0] = if cuda_atomic_add(cnt, h, 1) == ns - 1 { 1 } else { 0 }; }
    cuda_sync();
    if lb[0] == 0 { return; }
    var gm: f32 = -1e30;
    for u in ns { gm = max(gm, pm[(h * 16 + u) * 2]); }
    var gl: f32 = 0;
    var ga: f32 = 0;
    for u in ns {
        let f = exp(pm[(h * 16 + u) * 2] - gm);
        gl += pm[(h * 16 + u) * 2 + 1] * f;
        ga += pa[(h * 16 + u) * 128 + t] * f;
    }
    wq32(ga / gl, l, h * 4 + w, ao, aos, aom);
    if t == 0 { cnt[h] = 0; }
}
// o: the token and position; the token becomes the best, the position moves on.
cuda fn kamax(x: const f32[:], n: i64, o: i32[:]) {
    let t = cuda_thread();
    let bv = cuda_shared<f32>(32);
    let bi = cuda_shared<i32>(32);
    var mv: f32 = -3.0e38;
    var mi: i32 = 0;
    var i = t;
    while i < n { let v = x[i]; if v > mv { mv = v; mi = i as! i32; } i += 1024; }
    var m = 16;
    while m > 0 {
        let ov = cuda_shfl_xor(mv, m);
        let oi = cuda_shfl_xor(mi, m);
        if ov > mv || (ov == mv && oi < mi) { mv = ov; mi = oi; }
        m = m / 2;
    }
    if t % 32 == 0 { bv[t / 32] = mv; bi[t / 32] = mi; }
    cuda_sync();
    if t < 32 {
        mv = bv[t];
        mi = bi[t];
        m = 16;
        while m > 0 {
            let ov = cuda_shfl_xor(mv, m);
            let oi = cuda_shfl_xor(mi, m);
            if ov > mv || (ov == mv && oi < mi) { mv = ov; mi = oi; }
            m = m / 2;
        }
        if t == 0 { o[0] = mi; o[1] += 1; }
    }
}

// Q4_K's 6-bit scale and min of sub-block j, from the scale bytes at f[o..].
fn hksm(f: const u8[:], o: i64, j: i64) -> i64, i64 {
    if j < 4 { return (f[o + j] & 63) as i64, (f[o + j + 4] & 63) as i64; }
    return ((f[o + j + 4] & 15) | ((f[o + j - 4] >> 6) << 4)) as i64, ((f[o + j + 4] >> 4) | ((f[o + j] >> 6) << 4)) as i64;
}
// A Q4_K (fm 1) row's scales and mins a 32-block at f[p..], or a Q6_K (fm 2)
// row as i8 words, exactly, q - 32 with a scale a 16.
fn kunpack(f: const u8[:], fm: i64, p: i64, k: i64, u: i32[>..]&, sv: f32[>..]&, mv: f32[>..]&) {
    for sb in k / 256 {
        if fm == 1 {
            let bp = p + sb * 144;
            let d = HT[binary::u16le(f, bp)];
            let dm = HT[binary::u16le(f, bp + 2)];
            for j in 8 {
                let c, m = hksm(f, bp + 4, j);
                sv.push(d * c);
                mv.push(dm * m);
            }
        } else {
            let bp = p + sb * 210;
            let d = HT[binary::u16le(f, bp + 208)];
            for q in 16 { sv.push(d * (f[bp + 192 + q] as! i8)); }
            for j in 8 {
                let n = j / 4;
                let g = j % 4;
                for w in 8 {
                    var x: i32 = 0;
                    for i in 4 {
                        let l = 4 * w + i;
                        let lo = (f[bp + 64 * n + (g % 2) * 32 + l] >> (4 * (g / 2))) & 15;
                        let hi = (f[bp + 128 + 32 * n + l] >> (2 * g)) & 3;
                        x = x | ((((lo | (hi << 4)) as i32 - 32) & 255) << (8 * i));
                    }
                    u.push(x);
                }
            }
        }
    }
}
// For the batch kernels, 32 bytes a row of each 32-block (Q4_K: of each pair)
// of a 128-row block, 4 KB in a row: row i of block R is row R * 128 + i, or
// with pr gate row R * 64 + i, i < 64, then up row pr + R * 64 + i - 64.
fn grow(gr: i64, pr: i64) -> i64 {
    if pr == 0 { return gr; }
    let i = gr % 128;
    return if i < 64 { gr / 128 * 64 + i } else { pr + gr / 128 * 64 + i - 64 };
}
fn gtile(fm: i64, q: const i32[:], u: const i32[:], rows: i64, k: i64, pr: i64) -> i32[>..] {
    let kb = k / 32;
    let nc = if fm == 1 { kb / 2 } else { kb };
    var g: i32[>..] = [];
    g.resize(rows * nc * 8, 0);
    for gr in rows {
        let r = grow(gr, pr);
        for b in nc {
            let d = ((gr / 128 * nc + b) * 128 + gr % 128) * 8;
            for w in 8 {
                g[d + w] = if fm == 1 { q[(r * (kb / 8) + b / 4) * 36 + 4 + (b % 4) * 8 + w] } else if fm == 2 { u[r * (k / 4) + b * 8 + w] } else { q[r * (k / 4) + b * 8 + w] };
            }
        }
    }
    return g;
}
// Tensors ix stacked by rows: as stored where all are Q4_K or all Q6_K, else
// requantized to Q8_0. pr as for gtile, or -1 for no batch copy.
fn gmat(f: const u8[:], ts: const TI[:], base: i64, ix: const i64[:], pr: i64) -> GM {
    let k = ts[ix[0]].ne0;
    var ty = ts[ix[0]].ty;
    for i in ix { if ts[i].ty != ty { ty = -1; } }
    let fm = if k % 256 != 0 { 0 } else if ty == 12 { 1 } else if ty == 14 { 2 } else { 0 };
    var q: i32[>..] = [];
    var s: u16[>..] = [];
    var rows = 0;
    var t: Q8[>..] = [];
    for i in ix {
        let ti = ts[i];
        let rb = rbytes(ti.ty, k);
        for r in ti.ne1 {
            let p = base + ti.off + r * rb;
            if fm == 1 {
                for w in rb / 4 { q.push(binary::u32le(f, p + w * 4) as! i32); }
            } else if fm == 2 {
                for b in k / 256 {
                    let bp = p + b * 210;
                    for w in 52 { q.push(binary::u32le(f, bp + w * 4) as! i32); }
                    q.push(binary::u16le(f, bp + 208) as i32);
                }
            } else {
                t.clear();
                wcols(f, ti.ty, p, k, 0, k, t);
                for e in t {
                    s.push(e.d);
                    for g in 8 { q.push(((e.q[g * 4] as i32) & 255) | (((e.q[g * 4 + 1] as i32) & 255) << 8) | (((e.q[g * 4 + 2] as i32) & 255) << 16) | ((e.q[g * 4 + 3] as i32) << 24)); }
                }
            }
        }
        rows += ti.ne1;
    }
    // For the batch kernel: i8 weights (Q8_0's own), their scales and Q4_K's
    // mins as f32 by block then row; a block is 32 values, Q6_K's 16.
    var u: i32[>..] = [];
    var sv: f32[>..] = [];
    var mv: f32[>..] = [];
    for i in ix {
        let ti = ts[i];
        let rb = rbytes(ti.ty, k);
        for r in ti.ne1 { if fm != 0 && pr >= 0 { kunpack(f, fm, base + ti.off + r * rb, k, u, sv, mv); } }
    }
    let nper = if fm == 2 { k / 16 } else { k / 32 };
    var at: f32[>..] = [];
    var an: f32[>..] = [];
    var g: i32[>..] = [];
    if pr >= 0 {
        guard rows % 128 == 0 && k % 64 == 0 else { abort("--gpu: a matrix is not 128 rows by 64 columns a block"); }
        at.resize(rows * nper, 0);
        if fm == 1 { an.resize(rows * nper, 0); }
        for gr in rows {
            let r = grow(gr, pr);
            for b in nper {
                let d = (gr / 128 * nper + b) * 128 + gr % 128;
                at[d] = if fm == 0 { HT[s[r * nper + b]] } else { sv[r * nper + b] };
                if fm == 1 { an[d] = mv[r * nper + b]; }
            }
        }
        g = gtile(fm, q, u, rows, k, pr);
    }
    let m = GM { f: fm, q: cuda::alloc<i32>(q.len), s: cuda::alloc<u16>(max(s.len, 1)), u: cuda::alloc<i32>(max(g.len, 1)), t: cuda::alloc<f32>(max(at.len, 1)), n: cuda::alloc<f32>(max(an.len, 1)), rows: rows, k: k };
    cuda::upload(m.q, q);
    cuda::upload(m.s, s);
    cuda::upload(m.u, g);
    cuda::upload(m.t, at);
    cuda::upload(m.n, an);
    return m;
}
fn gfmt(ts: const TI[:], i: i64) -> i64 { if ts[i].ne0 % 256 != 0 { 0 } else if ts[i].ty == 12 { 1 } else if ts[i].ty == 14 { 2 } else { 0 } }
fn gload(f: const u8[:], ts: const TI[:], base: i64, te: i64) {
    guard cuda::init() else { abort("--gpu: no CUDA device"); }
    guard C.hd == 128 && !C.qkn && C.ctx <= 4096 && C.ctx % 64 == 0 else { abort("--gpu: needs head size 128, no q/k norm, -c <= 4096 and a multiple of 64"); }
    var nm: u8[>..] = [];
    cuda::device_name(nm);
    let hd = C.hd;
    let kvd = C.nkv * hd;
    let w = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"];
    var v: f32[>..] = [];
    for l in C.L {
        var id: i64[7] = [0; 7];
        for t in 7 { id[t] = tfind(f, ts, str("blk.", l, ".", w[t], ".weight")); guard id[t] >= 0 else { abort("missing tensor"); } }
        if gfmt(ts, id[0]) == gfmt(ts, id[2]) && ts[id[0]].ty == ts[id[2]].ty {
            GW.push(gmat(f, ts, base, [id[0], id[1], id[2]], 0));
            let e = GW[GW.len - 1];
            GW.push(GM { f: 0, q: e.q, s: e.s, u: e.u, t: e.t, n: e.n, rows: 0, k: 0 });
        } else {
            GW.push(gmat(f, ts, base, [id[0], id[1]], 0));
            GW.push(gmat(f, ts, base, [id[2]], 0));
        }
        GW.push(gmat(f, ts, base, [id[3]], 0));
        GW.push(gmat(f, ts, base, [id[4], id[5]], ts[id[4]].ne1));
        GW.push(gmat(f, ts, base, [id[6]], 0));
        GVO.push(v.len);
        vsl(f, ts, base, str("blk.", l, ".attn_norm.weight"), C.d, 0, C.d, v);
        GVO.push(v.len);
        vsl(f, ts, base, str("blk.", l, ".ffn_norm.weight"), C.d, 0, C.d, v);
        GVO.push(v.len);
        vsl(f, ts, base, str("blk.", l, ".attn_q.bias"), C.nh * hd, 0, C.nh * hd, v);
        vsl(f, ts, base, str("blk.", l, ".attn_k.bias"), kvd, 0, kvd, v);
        vsl(f, ts, base, str("blk.", l, ".attn_v.bias"), kvd, 0, kvd, v);
    }
    GVO.push(v.len);
    vsl(f, ts, base, "output_norm.weight", C.d, 0, C.d, v);
    let oi = tfind(f, ts, "output.weight");
    GW.push(gmat(f, ts, base, [if oi >= 0 { oi } else { te }], -1));
    GW.push(if oi >= 0 { gmat(f, ts, base, [te], -1) } else { GW[C.L * 5] });
    GV = cuda::alloc<f32>(v.len);
    cuda::upload(GV, v);
    var rt: f32[>..] = [];
    rtab(rt, C.ctx);
    GRT = cuda::alloc<f32>(rt.len);
    cuda::upload(GRT, rt);
    GX = cuda::alloc<f32>(C.d);
    GQKV = cuda::alloc<f32>(C.nh * hd + 2 * kvd);
    GPM = cuda::alloc<f32>(C.nh * 16 * 2);
    GPA = cuda::alloc<f32>(C.nh * 16 * hd);
    GLG = cuda::alloc<f32>(C.V);
    GKC = cuda::alloc<f32>(C.L * C.ctx * kvd);
    GVC = cuda::alloc<f32>(C.L * C.ctx * kvd);
    // A key tile can read up to 63 positions past the context: zeros, not NaNs.
    var zh: u16[>..] = [];
    zh.resize((C.L * C.ctx + 64) * kvd, 0);
    GKH = cuda::alloc<u16>(zh.len);
    GVT = cuda::alloc<u16>(zh.len);
    cuda::upload(GKH, zh);
    cuda::upload(GVT, zh);
    let mk = max(C.d, C.nh * hd);
    GXQ = cuda::alloc<i32>(mk / 4);
    GXS = cuda::alloc<f32>(mk / 32);
    GXM = cuda::alloc<f32>(mk / 32);
    GHQ = cuda::alloc<i32>(C.ff / 4);
    GHS = cuda::alloc<f32>(C.ff / 32);
    GHM = cuda::alloc<f32>(C.ff / 32);
    GO = cuda::alloc<i32>(2);
    GCNT = cuda::alloc<i32>(C.nh);
    var z: i32[>..] = [];
    z.resize(C.nh, 0);
    cuda::upload(GCNT, z);
    var nk = 0;
    for g in GW { if g.f != 0 { nk++; } }
    gballoc();
    // Load the kernels and capture both graphs now, outside any timing.
    var w0: i32[>..] = [];
    w0.resize(17, 0);
    var y: i32[>..] = [];
    w0.resize(257, 0);
    gbatch(w0[0..1], 0, y);
    gbatch(w0[0..17], 0, y);
    gbatch(w0[0..65], 0, y);
    gbatch(w0[0..129], 0, y);
    gbatch(w0, 0, y);
    gset(0, 0);
    gstep(true);
    cuda::sync();
    print(nm, " | ", cuda::memory() / 1048576, " MiB | CUDA, ", nk, "/", GW.len, " matrices in their k-quant layout");
}
fn gmv(m: GM, xq: cuda::buf<i32>, xs: cuda::buf<f32>, xm: cuda::buf<f32>, o: cuda::buf<f32>, bo: i64, ro: i64, mode: i64) {
    if m.rows == 0 { return; }
    let nb = (m.rows + 3) / 4;
    if m.f == 1 { cuda_launch(kmv1, nb, 128, m.q, m.s, xq, xs, xm, o, GV, bo, ro, m.rows, m.k, mode); }
    else if m.f == 2 { cuda_launch(kmv2, nb, 128, m.q, m.s, xq, xs, xm, o, GV, bo, ro, m.rows, m.k, mode); }
    else { cuda_launch(kmv0, nb, 128, m.q, m.s, xq, xs, xm, o, GV, bo, ro, m.rows, m.k, mode); }
}
fn gfwd() {
    let d = C.d;
    let hd = C.hd;
    let kvd = C.nkv * hd;
    let nq = C.nh * hd;
    if GP { cuda::sync(); GC = clock(); }
    let e = GW[C.L * 5 + 1];
    cuda_launch(kemb, (d + 255) / 256, 256, e.f, e.q, e.s, GX, GO, d);
    gt(0);
    let s0: f32 = 1 / sqrt(hd as f32);
    for l in C.L {
        cuda_launch(knq, 1, 512, GX, GV, GVO[l * 3], GXQ, GXS, GXM, d, C.eps);
        gt(1);
        let a = GW[l * 5];
        gmv(a, GXQ, GXS, GXM, GQKV, GVO[l * 3 + 2], 0, 2);
        gmv(GW[l * 5 + 1], GXQ, GXS, GXM, GQKV, GVO[l * 3 + 2], a.rows, 2);
        gt(2);
        let nr = (C.nh + 2 * C.nkv) * hd / 2;
        cuda_launch(krope, (nr + 127) / 128, 128, GQKV, GKC, GVC, GKH, GVT, GRT, GO, l * C.ctx, C.nh, C.nkv, hd, C.neox, C.ctx, l * kvd * C.ctx);
        gt(3);
        cuda_launch(katt, C.nh * 16, 128, GQKV, GKC, GVC, GPM, GPA, GO, l * C.ctx * kvd, kvd, C.nh / C.nkv, s0, GCNT, GXQ, GXS, GXM);
        gt(4);
        gt(5);
        gmv(GW[l * 5 + 2], GXQ, GXS, GXM, GX, 0, 0, 1);
        gt(6);
        cuda_launch(knq, 1, 512, GX, GV, GVO[l * 3 + 1], GXQ, GXS, GXM, d, C.eps);
        gt(1);
        let g = GW[l * 5 + 3];
        let gb = C.ff / 32;
        if g.f == 1 { cuda_launch(kgu1, gb, 1024, g.q, g.s, GXQ, GXS, GXM, GHQ, GHS, GHM, C.ff, g.k); }
        else if g.f == 2 { cuda_launch(kgu2, gb, 1024, g.q, g.s, GXQ, GXS, GXM, GHQ, GHS, GHM, C.ff, g.k); }
        else { cuda_launch(kgu0, gb, 512, g.q, g.s, GXQ, GXS, GXM, GHQ, GHS, GHM, C.ff, g.k); }
        gt(8);
        gmv(GW[l * 5 + 4], GHQ, GHS, GHM, GX, 0, 0, 1);
        gt(12);
    }
    cuda_launch(knq, 1, 512, GX, GV, GVO[C.L * 3], GXQ, GXS, GXM, d, C.eps);
    gt(1);
    gmv(GW[C.L * 5], GXQ, GXS, GXM, GLG, 0, 0, 0);
    gt(10);
}
// Batches of up to 16 tokens at positions st[1].., st[2] of them: each
// weight word loaded once for all. Token t's activations follow token
// t - 1's; outputs row r of token t at o[t * os + r].
inline fn mvt(f: i64, wq: const i32[:], ws: const u16[:], xq: const i32[:], xs: const f32[:], st: const i32[:], o: f32[:], os: i64, bias: const f32[:], bo: i64, ro: i64, rows: i64, k: i64, mode: i64) {
    let r = cuda_block() * 4 + cuda_thread() / 32;
    if r >= rows { return; }
    let nt: i64 = st[2];
    let l = cuda_thread() % 32;
    let k4 = k / 4;
    let kb = k / 32;
    var acc: f32[16] = [0.0; 16];
    if f == 0 {
        var j = l * 4;
        while j < k4 {
            let a0 = wq[r * k4 + j];
            let a1 = wq[r * k4 + j + 1];
            let a2 = wq[r * k4 + j + 2];
            let a3 = wq[r * k4 + j + 3];
            let sw = cuda_h2f(ws[r * kb + j / 8]);
            for t in 16 {
                if t < nt {
                    let xb = t * k4 + j;
                    let iv = cuda_dp4a(a3, xq[xb + 3], cuda_dp4a(a2, xq[xb + 2], cuda_dp4a(a1, xq[xb + 1], cuda_dp4a(a0, xq[xb], 0))));
                    acc[t] += iv * sw * xs[(t * kb + j / 8) * 2];
                }
            }
            j += 128;
        }
    } else if f == 1 {
        let nb = k / 256;
        var u = l;
        while u < nb * 8 {
            let blk = u / 8;
            let p = (u % 8) / 2;
            let hf = u % 2;
            let bo4 = (r * nb + blk) * 36;
            let hd = wq[bo4];
            let s0, n0 = ksm(wq[bo4 + 1], wq[bo4 + 2], wq[bo4 + 3], 2 * p);
            let s1, n1 = ksm(wq[bo4 + 1], wq[bo4 + 2], wq[bo4 + 3], 2 * p + 1);
            let wb = bo4 + 4 + p * 8 + hf * 4;
            let w0 = wq[wb];
            let w1 = wq[wb + 1];
            let w2 = wq[wb + 2];
            let w3 = wq[wb + 3];
            let d = cuda_h2f((hd & 65535) as! u16);
            let dm = cuda_h2f(((hd >> 16) & 65535) as! u16);
            let b0 = blk * 8 + 2 * p;
            for t in 16 {
                if t < nt {
                    let xb = t * k4 + blk * 64 + p * 16 + hf * 4;
                    let lo = cuda_dp4a(w3 & 0x0F0F0F0F, xq[xb + 3], cuda_dp4a(w2 & 0x0F0F0F0F, xq[xb + 2], cuda_dp4a(w1 & 0x0F0F0F0F, xq[xb + 1], cuda_dp4a(w0 & 0x0F0F0F0F, xq[xb], 0))));
                    let hi = cuda_dp4a((w3 >> 4) & 0x0F0F0F0F, xq[xb + 11], cuda_dp4a((w2 >> 4) & 0x0F0F0F0F, xq[xb + 10], cuda_dp4a((w1 >> 4) & 0x0F0F0F0F, xq[xb + 9], cuda_dp4a((w0 >> 4) & 0x0F0F0F0F, xq[xb + 8], 0))));
                    let sb = (t * kb + b0) * 2;
                    var v = d * (s0 * lo * xs[sb] + s1 * hi * xs[sb + 2]);
                    if hf == 0 { v -= dm * (n0 * xs[sb + 1] + n1 * xs[sb + 3]); }
                    acc[t] += v;
                }
            }
            u += 32;
        }
    } else {
        let nb = k / 256;
        var u = l;
        while u < nb * 16 {
            let blk = u / 16;
            let n = (u % 16) / 8;
            let l4 = u % 8;
            let bo6 = (r * nb + blk) * 53;
            let qa = wq[bo6 + 16 * n + l4];
            let qb = wq[bo6 + 16 * n + 8 + l4];
            let qh = wq[bo6 + 32 + 8 * n + l4];
            let v0 = (qa & 0x0F0F0F0F) | ((qh & 0x03030303) << 4);
            let v1 = (qb & 0x0F0F0F0F) | (((qh >> 2) & 0x03030303) << 4);
            let v2 = ((qa >> 4) & 0x0F0F0F0F) | (((qh >> 4) & 0x03030303) << 4);
            let v3 = ((qb >> 4) & 0x0F0F0F0F) | (((qh >> 6) & 0x03030303) << 4);
            let si = 8 * n + l4 / 4;
            let c0 = (wq[bo6 + 48 + si / 4] << (24 - (si % 4) * 8)) >> 24;
            let c1 = (wq[bo6 + 48 + (si + 2) / 4] << (24 - ((si + 2) % 4) * 8)) >> 24;
            let c2 = (wq[bo6 + 48 + (si + 4) / 4] << (24 - ((si + 4) % 4) * 8)) >> 24;
            let c3 = (wq[bo6 + 48 + (si + 6) / 4] << (24 - ((si + 6) % 4) * 8)) >> 24;
            let d = cuda_h2f((wq[bo6 + 52] & 65535) as! u16);
            let b0 = blk * 8 + n * 4;
            for t in 16 {
                if t < nt {
                    let xb = t * k4 + blk * 64 + n * 32 + l4;
                    let x0 = xq[xb];
                    let x1 = xq[xb + 8];
                    let x2 = xq[xb + 16];
                    let x3 = xq[xb + 24];
                    let sb = (t * kb + b0) * 2;
                    acc[t] += d * ((c0 * (cuda_dp4a(v0, x0, 0) - cuda_dp4a(0x20202020, x0, 0))) * xs[sb]
                                 + (c1 * (cuda_dp4a(v1, x1, 0) - cuda_dp4a(0x20202020, x1, 0))) * xs[sb + 2]
                                 + (c2 * (cuda_dp4a(v2, x2, 0) - cuda_dp4a(0x20202020, x2, 0))) * xs[sb + 4]
                                 + (c3 * (cuda_dp4a(v3, x3, 0) - cuda_dp4a(0x20202020, x3, 0))) * xs[sb + 6]);
                }
            }
            u += 32;
        }
    }
    for t in 16 {
        if t < nt {
            let v = wsum(acc[t]);
            if l == 0 {
                let y = t * os + ro + r;
                if mode == 1 { o[y] += v; } else if mode == 2 { o[y] = v + bias[bo + ro + r]; } else { o[y] = v; }
            }
        }
    }
}
cuda fn kmt0(wq: const i32[:], ws: const u16[:], xq: const i32[:], xs: const f32[:], st: const i32[:], o: f32[:], os: i64, bias: const f32[:], bo: i64, ro: i64, rows: i64, k: i64, mode: i64) { mvt(0, wq, ws, xq, xs, st, o, os, bias, bo, ro, rows, k, mode); }
cuda fn kmt1(wq: const i32[:], ws: const u16[:], xq: const i32[:], xs: const f32[:], st: const i32[:], o: f32[:], os: i64, bias: const f32[:], bo: i64, ro: i64, rows: i64, k: i64, mode: i64) { mvt(1, wq, ws, xq, xs, st, o, os, bias, bo, ro, rows, k, mode); }
cuda fn kmt2(wq: const i32[:], ws: const u16[:], xq: const i32[:], xs: const f32[:], st: const i32[:], o: f32[:], os: i64, bias: const f32[:], bo: i64, ro: i64, rows: i64, k: i64, mode: i64) { mvt(2, wq, ws, xq, xs, st, o, os, bias, bo, ro, rows, k, mode); }
// Token cuda_block() / per of the batch: tokens from tk.
cuda fn kembt(f: i64, q: const i32[:], s: const u16[:], x: f32[:], tk: const i32[:], st: const i32[:], k: i64) {
    let per = (k + 255) / 256;
    let t = cuda_block() / per;
    if t >= st[2] { return; }
    let j = (cuda_block() % per) * 256 + cuda_thread();
    if j >= k { return; }
    let row: i64 = tk[t];
    let y = t * k + j;
    if f == 0 {
        let w = q[row * (k / 4) + j / 4];
        x[y] = cuda_h2f(s[row * (k / 32) + j / 32]) * ((w << ((3 - j % 4) * 8)) >> 24);
        return;
    }
    let nb = k / 256;
    let w = j % 256;
    if f == 1 {
        let bo = (row * nb + j / 256) * 36;
        let js = w / 32;
        let l = w % 32;
        let sc, mn = ksm(q[bo + 1], q[bo + 2], q[bo + 3], js);
        let v = (q[bo + 4 + (js / 2) * 8 + l / 4] >> ((l % 4) * 8 + (js % 2) * 4)) & 15;
        let hd = q[bo];
        x[y] = cuda_h2f((hd & 65535) as! u16) * (sc * v) - cuda_h2f(((hd >> 16) & 65535) as! u16) * mn;
        return;
    }
    let bo = (row * nb + j / 256) * 53;
    let n = w / 128;
    let g = (w % 128) / 32;
    let l = w % 32;
    let lo = (q[bo + 16 * n + (g % 2) * 8 + l / 4] >> ((l % 4) * 8 + (g / 2) * 4)) & 15;
    let hi = (q[bo + 32 + 8 * n + l / 4] >> ((l % 4) * 8 + 2 * g)) & 3;
    let si = 8 * n + l / 16 + 2 * g;
    let sc = (q[bo + 48 + si / 4] << (24 - (si % 4) * 8)) >> 24;
    x[y] = cuda_h2f((q[bo + 52] & 65535) as! u16) * (sc * ((lo | (hi << 4)) - 32));
}
// Token cuda_block()'s x normed by w[wo..] and quantized.
cuda fn knqt(x: const f32[:], w: const f32[:], wo: i64, o: i32[:], os: f32[:], st: const i32[:], d: i64, eps: f32, big: bool) {
    let tb = cuda_block();
    if tb >= st[2] { return; }
    let t = cuda_thread();
    let bd = cuda_block_dim();
    let sh = cuda_shared<f32>(32);
    let xo = tb * d;
    var s: f32 = 0;
    var i = t;
    while i < d { s += x[xo + i] * x[xo + i]; i += bd; }
    s = wsum(s);
    if t % 32 == 0 { sh[t / 32] = s; }
    cuda_sync();
    var tot: f32 = 0;
    for k in bd / 32 { tot += sh[k]; }
    let sc: f32 = 1 / sqrt(tot / d + eps);
    i = t;
    while i < d { wq32t(x[xo + i] * sc * w[wo + i], t % 32, tb, i / 32, d / 32, big, o, os); i += bd; }
}
// q and k of every token rotated at its own position, k and v to the caches.
cuda fn kropet(qkv: f32[:], kc: f32[:], vc: f32[:], kh: u16[:], vt: u16[:], rt: const f32[:], st: const i32[:], lc: i64, nh: i64, nkv: i64, neox: bool, ctx: i64, lv: i64) {
    let per = (nh + 2 * nkv) * 64;
    let i0 = cuda_block() * cuda_block_dim() + cuda_thread();
    let tb = i0 / per;
    if tb >= st[2] { return; }
    let i = i0 % per;
    let pos: i64 = st[1] + tb;
    let qs = (nh + 2 * nkv) * 128;
    let qb = tb * qs;
    let co = (lc + pos) * nkv * 128;
    let nq = nh * 64;
    let nk = nkv * 64;
    if i >= nq + nk {
        let p = (i - nq - nk) * 2;
        let src = qb + nh * 128 + nkv * 128 + p;
        vc[co + p] = qkv[src];
        vc[co + p + 1] = qkv[src + 1];
        vt[lv + p * ctx + pos] = cuda_f2h(qkv[src]);
        vt[lv + (p + 1) * ctx + pos] = cuda_f2h(qkv[src + 1]);
        return;
    }
    let hh = i / 64;
    let e = i % 64;
    let ia = qb + (if neox { hh * 128 + e } else { hh * 128 + 2 * e });
    let ib = if neox { ia + 64 } else { ia + 1 };
    let cs = rt[pos * 128 + 2 * e];
    let sn = rt[pos * 128 + 2 * e + 1];
    let a = qkv[ia];
    let b = qkv[ib];
    let ra = a * cs - b * sn;
    let rb = a * sn + b * cs;
    qkv[ia] = ra;
    qkv[ib] = rb;
    if hh >= nh {
        kc[co + ia - qb - nh * 128] = ra;
        kc[co + ib - qb - nh * 128] = rb;
        kh[co + ia - qb - nh * 128] = cuda_f2h(ra);
        kh[co + ib - qb - nh * 128] = cuda_f2h(rb);
    }
}
// katt for token t = cuda_block() / (nh * 16) of the batch, which sees the
// positions up to its own.
cuda fn katt_t(qkv: const f32[:], kc: const f32[:], vc: const f32[:], pm: f32[:], pa: f32[:], st: const i32[:], lo: i64, kvd: i64, nh: i64, gq: i64, s0: f32, cnt: i32[:], ao: i32[:], aos: f32[:]) {
    let tb = cuda_block() / (nh * 4);
    if tb >= st[2] { return; }
    let n: i64 = st[1] + tb + 1;
    let cs = max(32, (n + 3) / 4);
    let bi = cuda_block() % (nh * 4);
    let h = bi / 4;
    let c = bi % 4;
    if c * cs >= n { return; }
    let t = cuda_thread();
    let w = t / 32;
    let l = t % 32;
    let kb = lo + (h / gq) * 128 + l * 4;
    let qo = tb * (nh * 128 + 2 * kvd) + h * 128 + l * 4;
    let q0 = qkv[qo] * s0;
    let q1 = qkv[qo + 1] * s0;
    let q2 = qkv[qo + 2] * s0;
    let q3 = qkv[qo + 3] * s0;
    var m: f32 = -1e30;
    var sum: f32 = 0;
    var o0: f32 = 0;
    var o1: f32 = 0;
    var o2: f32 = 0;
    var o3: f32 = 0;
    let a1 = min(c * cs + cs, n);
    var s = c * cs + w;
    // Four positions a step: their dot products reduce side by side and the
    // running max moves once.
    while s + 12 < a1 {
        let ka = kb + s * kvd;
        let kc4 = kb + (s + 4) * kvd;
        let kd = kb + (s + 8) * kvd;
        let ke = kb + (s + 12) * kvd;
        let pa0 = wsum(q0 * kc[ka] + q1 * kc[ka + 1] + q2 * kc[ka + 2] + q3 * kc[ka + 3]);
        let pa1 = wsum(q0 * kc[kc4] + q1 * kc[kc4 + 1] + q2 * kc[kc4 + 2] + q3 * kc[kc4 + 3]);
        let pa2 = wsum(q0 * kc[kd] + q1 * kc[kd + 1] + q2 * kc[kd + 2] + q3 * kc[kd + 3]);
        let pa3 = wsum(q0 * kc[ke] + q1 * kc[ke + 1] + q2 * kc[ke + 2] + q3 * kc[ke + 3]);
        let mn = max(max(m, max(pa0, pa1)), max(pa2, pa3));
        let f = exp(m - mn);
        let e0 = exp(pa0 - mn);
        let e1 = exp(pa1 - mn);
        let e2 = exp(pa2 - mn);
        let e3 = exp(pa3 - mn);
        sum = sum * f + ((e0 + e1) + (e2 + e3));
        o0 = o0 * f + ((e0 * vc[ka] + e1 * vc[kc4]) + (e2 * vc[kd] + e3 * vc[ke]));
        o1 = o1 * f + ((e0 * vc[ka + 1] + e1 * vc[kc4 + 1]) + (e2 * vc[kd + 1] + e3 * vc[ke + 1]));
        o2 = o2 * f + ((e0 * vc[ka + 2] + e1 * vc[kc4 + 2]) + (e2 * vc[kd + 2] + e3 * vc[ke + 2]));
        o3 = o3 * f + ((e0 * vc[ka + 3] + e1 * vc[kc4 + 3]) + (e2 * vc[kd + 3] + e3 * vc[ke + 3]));
        m = mn;
        s += 16;
    }
    while s < a1 {
        let ko = kb + s * kvd;
        let p = wsum(q0 * kc[ko] + q1 * kc[ko + 1] + q2 * kc[ko + 2] + q3 * kc[ko + 3]);
        let mn = max(m, p);
        let f = exp(m - mn);
        let e = exp(p - mn);
        sum = sum * f + e;
        o0 = o0 * f + e * vc[ko];
        o1 = o1 * f + e * vc[ko + 1];
        o2 = o2 * f + e * vc[ko + 2];
        o3 = o3 * f + e * vc[ko + 3];
        m = mn;
        s += 4;
    }
    let shm = cuda_shared<f32>(8);
    let sho = cuda_shared<f32>(512);
    if l == 0 { shm[w] = m; shm[4 + w] = sum; }
    sho[w * 128 + l * 4] = o0;
    sho[w * 128 + l * 4 + 1] = o1;
    sho[w * 128 + l * 4 + 2] = o2;
    sho[w * 128 + l * 4 + 3] = o3;
    cuda_sync();
    let mx = max(max(shm[0], shm[1]), max(shm[2], shm[3]));
    var sl: f32 = 0;
    var sa: f32 = 0;
    for k in 4 { let f = exp(shm[k] - mx); sl += shm[4 + k] * f; sa += sho[k * 128 + t] * f; }
    let pi = (tb * nh + h) * 16 + c;
    pa[pi * 128 + t] = sa;
    if t == 0 { pm[pi * 2] = mx; pm[pi * 2 + 1] = sl; }
    cuda_fence();
    let lb = cuda_shared<i32>(1);
    let ns = (n + cs - 1) / cs;
    let ci = tb * nh + h;
    if t == 0 { lb[0] = if cuda_atomic_add(cnt, ci, 1) == ns - 1 { 1 } else { 0 }; }
    cuda_sync();
    if lb[0] == 0 { return; }
    let pb = (tb * nh + h) * 16;
    var gm: f32 = -1e30;
    for u in ns { gm = max(gm, pm[(pb + u) * 2]); }
    var gl: f32 = 0;
    var ga: f32 = 0;
    for u in ns {
        let f = exp(pm[(pb + u) * 2] - gm);
        gl += pm[(pb + u) * 2 + 1] * f;
        ga += pa[(pb + u) * 128 + t] * f;
    }
    wq32t(ga / gl, l, tb, h * 4 + w, nh * 4, false, ao, aos);
    if t == 0 { cnt[ci] = 0; }
}
// Flash attention for a batch on tensor cores, in halves: block (h << tt) + qt
// takes tokens qt * 64.. of head h, a warp 16 of them, against 64 keys at a
// time through shared memory (cp.async from GKH, GVT), K by key (rows of 68
// words) and V by dimension (36), the softmax online; the output quantized
// as wq32t lays it out.
cuda fn kfa(qkv: const f32[:], kh: const i32[:], vt: const i32[:], st: const i32[:], lo: i64, kvd: i64, nh: i64, gq: i64, s0: f32, tt: i64, ctx: i64, lv: i64, ao: i32[:], aos: f32[:]) {
    let nt: i64 = st[2];
    let p0: i64 = st[1];
    let h = cuda_block() >> tt;
    let q0 = (cuda_block() & ((1 << tt) - 1)) * 64;
    if q0 >= nt { return; }
    let th = cuda_thread();
    let w = (th / 32) as! i32;
    let l = (th % 32) as! i32;
    let g = l / 4;
    let t = l % 4;
    let qs = nh * 128 + 2 * kvd;
    let kb = lo + (h / gq) * 128;
    let vb = lv + (h / gq) * 128 * ctx;
    let ksm = cuda_shared<i32>(4352);
    let vsm = cuda_shared<i32>(4608);
    let ra = q0 + (w * 16 + g) as i64;
    let rb = ra + 8;
    let qa = min(ra, nt - 1) * qs + h * 128 + (t * 2) as i64;
    let qb = min(rb, nt - 1) * qs + h * 128 + (t * 2) as i64;
    var qf: i32[32] = [0; 32];
    for ks in 8 {
        let c = ks * 16;
        qf[ks * 4] = cuda_f2h2(qkv[qa + c] * s0, qkv[qa + c + 1] * s0);
        qf[ks * 4 + 1] = cuda_f2h2(qkv[qb + c] * s0, qkv[qb + c + 1] * s0);
        qf[ks * 4 + 2] = cuda_f2h2(qkv[qa + c + 8] * s0, qkv[qa + c + 9] * s0);
        qf[ks * 4 + 3] = cuda_f2h2(qkv[qb + c + 8] * s0, qkv[qb + c + 9] * s0);
    }
    let pa = p0 + ra;
    let pb = p0 + rb;
    let nk = p0 + min(q0 + 64, nt);
    let wl = p0 + min(q0 + (w * 16 + 15) as i64, nt - 1);
    var o: f32[64] = [0.0; 64];
    var ma: f32 = -1e30;
    var mb: f32 = -1e30;
    var la: f32 = 0;
    var lb: f32 = 0;
    var k0: i64 = 0;
    while k0 < nk {
        cuda_sync();
        for i in 8 {
            let e = th + i * 128;
            cuda_cp_async(ksm, (e / 16) * 68 + (e % 16) * 4, kh, (kb + (k0 + e / 16) * kvd) / 2 + (e % 16) * 4);
            cuda_cp_async(vsm, (e / 8) * 36 + (e % 8) * 4, vt, (vb + (e / 8) * ctx + k0) / 2 + (e % 8) * 4);
        }
        cuda_cp_commit();
        cuda_cp_wait(0);
        cuda_sync();
        if k0 <= wl {
            var sc: f32[32] = [0.0; 32];
            for n in 8 {
                var c: f32[4] = [0.0, 0.0, 0.0, 0.0];
                let kr = ((n as! i32) * 8 + g) * 68 + t;
                for ks in 8 {
                    let fa: i32[4] = [qf[ks * 4], qf[ks * 4 + 1], qf[ks * 4 + 2], qf[ks * 4 + 3]];
                    let fb: i32[2] = [ksm[kr + (ks as! i32) * 8], ksm[kr + (ks as! i32) * 8 + 4]];
                    c = cuda_mmah(fa, fb, c);
                }
                let kp = k0 + n * 8 + (t * 2) as i64;
                sc[n * 4] = if kp <= pa { c[0] } else { -1e30 };
                sc[n * 4 + 1] = if kp < pa { c[1] } else { -1e30 };
                sc[n * 4 + 2] = if kp <= pb { c[2] } else { -1e30 };
                sc[n * 4 + 3] = if kp < pb { c[3] } else { -1e30 };
            }
            var xa = ma;
            var xb = mb;
            for n in 8 {
                xa = max(xa, max(sc[n * 4], sc[n * 4 + 1]));
                xb = max(xb, max(sc[n * 4 + 2], sc[n * 4 + 3]));
            }
            xa = max(xa, cuda_shfl_xor(xa, 1));
            xa = max(xa, cuda_shfl_xor(xa, 2));
            xb = max(xb, cuda_shfl_xor(xb, 1));
            xb = max(xb, cuda_shfl_xor(xb, 2));
            let fa = exp(ma - xa);
            let fb = exp(mb - xb);
            ma = xa;
            mb = xb;
            la *= fa;
            lb *= fb;
            for dn in 16 {
                o[dn * 4] *= fa;
                o[dn * 4 + 1] *= fa;
                o[dn * 4 + 2] *= fb;
                o[dn * 4 + 3] *= fb;
            }
            // P as the A fragments of the 4 key steps: n-tiles 2 kk and 2 kk + 1.
            var pf: i32[16] = [0; 16];
            for n in 8 {
                let e0 = exp(sc[n * 4] - ma);
                let e1 = exp(sc[n * 4 + 1] - ma);
                let e2 = exp(sc[n * 4 + 2] - mb);
                let e3 = exp(sc[n * 4 + 3] - mb);
                la += e0 + e1;
                lb += e2 + e3;
                pf[(n / 2) * 4 + (n % 2) * 2] = cuda_f2h2(e0, e1);
                pf[(n / 2) * 4 + (n % 2) * 2 + 1] = cuda_f2h2(e2, e3);
            }
            for dn in 16 {
                var c: f32[4] = [o[dn * 4], o[dn * 4 + 1], o[dn * 4 + 2], o[dn * 4 + 3]];
                let vr = ((dn as! i32) * 8 + g) * 36 + t;
                for kk in 4 {
                    let fa: i32[4] = [pf[kk * 4], pf[kk * 4 + 1], pf[kk * 4 + 2], pf[kk * 4 + 3]];
                    let fb: i32[2] = [vsm[vr + (kk as! i32) * 8], vsm[vr + (kk as! i32) * 8 + 4]];
                    c = cuda_mmah(fa, fb, c);
                }
                o[dn * 4] = c[0];
                o[dn * 4 + 1] = c[1];
                o[dn * 4 + 2] = c[2];
                o[dn * 4 + 3] = c[3];
            }
        }
        k0 += 64;
    }
    la += cuda_shfl_xor(la, 1);
    la += cuda_shfl_xor(la, 2);
    lb += cuda_shfl_xor(lb, 1);
    lb += cuda_shfl_xor(lb, 2);
    // A row's 32-dim block is 8 values in each lane of its quad: 2 a n-tile.
    let kq = nh * 4;
    for bk in 4 {
        for hr in 2 {
            let inv = 1 / (if hr == 0 { la } else { lb });
            var v: f32[8] = [0.0; 8];
            for j in 4 {
                v[j * 2] = o[(bk * 4 + j) * 4 + hr * 2] * inv;
                v[j * 2 + 1] = o[(bk * 4 + j) * 4 + hr * 2 + 1] * inv;
            }
            var am: f32 = 0;
            for i in 8 { am = max(am, abs(v[i])); }
            am = max(am, cuda_shfl_xor(am, 1));
            am = max(am, cuda_shfl_xor(am, 2));
            let id: f32 = if am != 0 { 127 / am } else { 0 };
            var qsum: f32 = 0;
            var wd: i32[4] = [0, 0, 0, 0];
            for j in 4 {
                let u0 = round_even(v[j * 2] * id) as! i32;
                let u1 = round_even(v[j * 2 + 1] * id) as! i32;
                qsum += (u0 + u1) as f32;
                let hw = ((u0 & 255) | ((u1 & 255) << 8)) << ((t % 2) * 16);
                wd[j] = hw | cuda_shfl_xor(hw, 1);
            }
            qsum += cuda_shfl_xor(qsum, 1);
            qsum += cuda_shfl_xor(qsum, 2);
            let tk = if hr == 0 { ra } else { rb };
            if tk < nt {
                let blk = h * 4 + bk;
                if t % 2 == 0 { for j in 4 { ao[(tk * kq + blk) * 8 + j * 2 + (t / 2) as i64] = wd[j]; } }
                if t == 0 { let i = bsi(tk, blk); aos[i] = am / 127; aos[i + 1024] = am / 127 * qsum; }
            }
        }
    }
}
// silu(gate) * up of every token, quantized a warp per 32.
cuda fn kswt(gu: const f32[:], st: const i32[:], o: i32[:], os: f32[:], ff: i64) {
    let b = cuda_block() * 4 + cuda_thread() / 32;
    let per = ff / 32;
    let tb = b / per;
    if tb >= st[2] { return; }
    let l = cuda_thread() % 32;
    let j = (b % per) * 32 + l;
    let g = gu[tb * 2 * ff + j];
    wq32t(g / (1 + exp(-g)) * gu[tb * 2 * ff + ff + j], l, tb, b % per, per, false, o, os);
}
// The best of each token's logits, to out.
cuda fn kamaxt(x: const f32[:], n: i64, st: const i32[:], out: i32[:]) {
    let tb = cuda_block();
    if tb >= st[2] { return; }
    let t = cuda_thread();
    let bv = cuda_shared<f32>(32);
    let bi = cuda_shared<i32>(32);
    var mv: f32 = -3.0e38;
    var mi: i32 = 0;
    var i = t;
    while i < n { let v = x[tb * n + i]; if v > mv { mv = v; mi = i as! i32; } i += 1024; }
    var m = 16;
    while m > 0 {
        let ov = cuda_shfl_xor(mv, m);
        let oi = cuda_shfl_xor(mi, m);
        if ov > mv || (ov == mv && oi < mi) { mv = ov; mi = oi; }
        m = m / 2;
    }
    if t % 32 == 0 { bv[t / 32] = mv; bi[t / 32] = mi; }
    cuda_sync();
    if t < 32 {
        mv = bv[t];
        mi = bi[t];
        m = 16;
        while m > 0 {
            let ov = cuda_shfl_xor(mv, m);
            let oi = cuda_shfl_xor(mi, m);
            if ov > mv || (ov == mv && oi < mi) { mv = ov; mi = oi; }
            m = m / 2;
        }
        if t == 0 { out[tb] = mi; }
    }
}
// Q4_K, 128 rows a block: a stage is a pair of 32-blocks, whose 32 bytes a row
// (gtile's) hold both as low and high nibbles; scales and mins (at, an) by
// 128-row block, then block, then row. A warp's 32 rows are two A tiles
// against each B fragment.
fn q4stage(wq: const i32[:], at: const f32[:], an: const f32[:], xq: const i32[:], xs: const f32[:], sa: i32[:], sb: i32[:], sw: f32[:], sn: f32[:], sx: f32[:], p: i64, s: i64, r0: i64, t0: i64, rows: i64, nt: i64, k4: i64, kb: i64, pr: i64) {
    let th = cuda_thread();
    for i in 2 {
        let e = th + i * 128;
        cuda_cp_async(sa, s * 1536 + (e / 2) * 12 + (e % 2) * 4, wq, ((r0 / 128 * (kb / 2) + p) * 128 + e / 2) * 8 + (e % 2) * 4);
    }
    for i in 2 {
        let e = th * 2 + i;
        cuda_cp_async(sb, s * 1280 + (e / 4) * 20 + (e % 4) * 4, xq, min(t0 + e / 4, nt - 1) * k4 + p * 16 + (e % 4) * 4);
    }
    if th < 64 { cuda_cp_async(sw, s * 256 + (th / 32) * 128 + (th % 32) * 4, at, (r0 / 128 * kb + p * 2 + th / 32) * 128 + (th % 32) * 4); }
    else { cuda_cp_async(sn, s * 256 + ((th - 64) / 32) * 128 + (th % 32) * 4, an, (r0 / 128 * kb + p * 2 + (th - 64) / 32) * 128 + (th % 32) * 4); }
    if th < 64 { cuda_cp_async(sx, s * 256 + th * 4, xs, p * 2048 + (th / 32) * 1024 + t0 * 2 + (th % 32) * 4); }
}
cuda fn kmq4(wq: const i32[:], at: const f32[:], an: const f32[:], xq: const i32[:], xs: const f32[:], st: const i32[:], o: f32[:], os: i64, bias: const f32[:], bo: i64, ro: i64, rows: i64, k: i64, mode: i64, sp: i64, tt: i64, pt: f32[:], pr: i64, hq: i32[:], hqs: f32[:]) {
    let nt: i64 = st[2];
    let nbn = (nt + 63) / 64;
    let cb = (cuda_block() >> tt) as! i32;
    let s4 = sp as! i32;
    let bm = (cb / s4) as i64;
    let ks = (cb % s4) as i64;
    let bn = cuda_block() & ((1 << tt) - 1);
    if bn >= nbn || bm * 128 >= rows { return; }
    let w = (cuda_thread() / 32) as! i32;
    let l = (cuda_thread() % 32) as! i32;
    let g = l / 4;
    let t = l % 4;
    let k4 = k / 4;
    let kb = k / 32;
    let r0 = bm * 128;
    let t0 = bn * 64;
    let sa = cuda_shared<i32>(3072);
    let sb = cuda_shared<i32>(2560);
    let sw = cuda_shared<f32>(512);
    let sn = cuda_shared<f32>(512);
    let sx = cuda_shared<f32>(512);
    var acc: f32[64] = [0.0; 64];
    let z: i32[4] = [0, 0, 0, 0];
    let np = kb / 2;
    let p0 = ((ks as! i32) * (np as! i32) / s4) as i64;
    let p1 = ((ks as! i32 + 1) * (np as! i32) / s4) as i64;
    if p0 < p1 { q4stage(wq, at, an, xq, xs, sa, sb, sw, sn, sx, p0, 0, r0, t0, rows, nt, k4, kb, pr); }
    cuda_cp_commit();
    // ldmatrix: lane l gives row l % 8 of matrix l / 8.
    let mi = l / 8;
    let ri = l % 8;
    let la = (w * 32 + (mi % 2) * 8 + ri) * 12 + (mi / 2) * 4;
    let lb = ((mi / 2) * 8 + ri) * 20 + (mi % 2) * 4;
    var s: i32 = 0;
    for p in p0..p1 {
        cuda_cp_wait(0);
        cuda_sync();
        if p + 1 < p1 { q4stage(wq, at, an, xq, xs, sa, sb, sw, sn, sx, p + 1, 1 - s as i64, r0, t0, rows, nt, k4, kb, pr); }
        cuda_cp_commit();
        let wa = cuda_ldsm4(sa, s * 1536 + la);
        let va = cuda_ldsm4(sa, s * 1536 + la + 192);
        let w0 = wa[0];
        let w1 = wa[1];
        let w2 = wa[2];
        let w3 = wa[3];
        let v0 = va[0];
        let v1 = va[1];
        let v2 = va[2];
        let v3 = va[3];
        for jj in 2 {
            let sh = (jj as! i32) * 4;
            let fa: i32[4] = [(w0 >> sh) & 0x0F0F0F0F, (w1 >> sh) & 0x0F0F0F0F, (w2 >> sh) & 0x0F0F0F0F, (w3 >> sh) & 0x0F0F0F0F];
            let fc: i32[4] = [(v0 >> sh) & 0x0F0F0F0F, (v1 >> sh) & 0x0F0F0F0F, (v2 >> sh) & 0x0F0F0F0F, (v3 >> sh) & 0x0F0F0F0F];
            let wo = s * 256 + (jj as! i32) * 128 + w * 32 + g;
            let da = sw[wo];
            let db = sw[wo + 8];
            let dc = sw[wo + 16];
            let dd = sw[wo + 24];
            let na = sn[wo];
            let nb = sn[wo + 8];
            let nc = sn[wo + 16];
            let nd = sn[wo + 24];
            for np in 4 {
                let bb = cuda_ldsm4(sb, s * 1280 + (np as! i32) * 320 + lb + (jj as! i32) * 8);
                for hf in 2 {
                    let n = np * 2 + hf;
                    let fb: i32[2] = [bb[hf * 2], bb[hf * 2 + 1]];
                    let c = cuda_mma(fa, fb, z);
                    let e = cuda_mma(fc, fb, z);
                    let xo = s * 256 + ((n as! i32) * 8 + t * 2) * 2 + (jj as! i32);
                    let s0 = sx[xo];
                    let s1 = sx[xo + 2];
                    let m0 = sx[xo + 128];
                    let m1 = sx[xo + 130];
                    acc[n * 8] = mul_add(-na, m0, mul_add(c[0] as f32, da * s0, acc[n * 8]));
                    acc[n * 8 + 1] = mul_add(-na, m1, mul_add(c[1] as f32, da * s1, acc[n * 8 + 1]));
                    acc[n * 8 + 2] = mul_add(-nb, m0, mul_add(c[2] as f32, db * s0, acc[n * 8 + 2]));
                    acc[n * 8 + 3] = mul_add(-nb, m1, mul_add(c[3] as f32, db * s1, acc[n * 8 + 3]));
                    acc[n * 8 + 4] = mul_add(-nc, m0, mul_add(e[0] as f32, dc * s0, acc[n * 8 + 4]));
                    acc[n * 8 + 5] = mul_add(-nc, m1, mul_add(e[1] as f32, dc * s1, acc[n * 8 + 5]));
                    acc[n * 8 + 6] = mul_add(-nd, m0, mul_add(e[2] as f32, dd * s0, acc[n * 8 + 6]));
                    acc[n * 8 + 7] = mul_add(-nd, m1, mul_add(e[3] as f32, dd * s1, acc[n * 8 + 7]));
                }
            }
        }
        s = 1 - s;
    }
    if pr > 0 {
        // silu(gate) * up an n-tile at a time, gate rows through sw and up rows
        // through sn; a token's 32 pairs one quantized block of h.
        const for <N> in [0, 1, 2, 3, 4, 5, 6, 7] {
            cuda_sync();
            const for <I> in [0, 1, 2, 3, 4, 5, 6, 7] {
                let lr = w * 32 + g + (I / 2) * 8;
                let tl = t * 2 + I % 2;
                if w < 2 { sw[tl * 64 + lr] = acc[N * 8 + I]; } else { sn[tl * 64 + lr - 64] = acc[N * 8 + I]; }
            }
            cuda_sync();
            for j in 2 {
                let tl = w * 2 + j as! i32;
                let tk = t0 + N * 8 + tl as i64;
                if tk < nt {
                    for q in 2 {
                        let gv = sw[tl * 64 + (q as! i32) * 32 + l];
                        wq32t(gv / (1 + exp(-gv)) * sn[tl * 64 + (q as! i32) * 32 + l], l as i64, tk, r0 / 64 + q, pr / 32, true, hq, hqs);
                    }
                }
            }
        }
        return;
    }
    let rr = r0 + (w * 32 + g) as i64;
    const for <N> in [0, 1, 2, 3, 4, 5, 6, 7] {
        const for <I> in [0, 1, 2, 3, 4, 5, 6, 7] {
            let r = rr + (I / 2) * 8;
            let tk = t0 + N * 8 + (t * 2) as i64 + I % 2;
            if r < rows && tk < nt {
                let v = acc[N * 8 + I];
                if sp > 1 { pt[(ks * 512 + tk) * rows + r] = v; }
                else {
                    let y = tk * os + ro + r;
                    if mode == 1 { o[y] += v; } else if mode == 2 { o[y] = v + bias[bo + ro + r]; } else { o[y] = v; }
                }
            }
        }
    }
}
// Q8_0 two 32-blocks a stage: a row's 64 bytes are 4 chunks of 16, chunk c of
// row r at c ^ ((r >> 1) & 3), which keeps the fragment loads off each other's
// banks with no padding; the scales of both blocks (at) and the tokens' (xs,
// as bsi lays them out) beside them. Rows as gtile lays them out.
fn q8stage(wq: const i32[:], at: const f32[:], xq: const i32[:], xs: const f32[:], sa: i32[:], sb: i32[:], sw: f32[:], sx: f32[:], p: i64, s: i64, r0: i64, t0: i64, nt: i64, k4: i64, kb: i64) {
    let th = cuda_thread();
    for i in 4 {
        let e = th + i * 128;
        let r = e / 4;
        let c = e % 4;
        cuda_cp_async(sa, s * 2048 + r * 16 + ((c ^ ((r / 2) % 4)) * 4), wq, ((r0 / 128 * kb + p * 2 + c / 2) * 128 + r) * 8 + (c % 2) * 4);
    }
    for i in 2 {
        let e = th + i * 128;
        let tk = e / 4;
        let c = e % 4;
        cuda_cp_async(sb, s * 1024 + tk * 16 + ((c ^ ((tk / 2) % 4)) * 4), xq, min(t0 + tk, nt - 1) * k4 + p * 16 + c * 4);
    }
    if th < 64 { cuda_cp_async(sw, s * 256 + th * 4, at, (r0 / 128 * kb + p * 2) * 128 + th * 4); }
    else if th < 96 { cuda_cp_async(sx, s * 128 + (th - 64) * 4, xs, p * 2048 + t0 * 2 + (th - 64) * 4); }
}
cuda fn kmq8(wq: const i32[:], at: const f32[:], xq: const i32[:], xs: const f32[:], st: const i32[:], o: f32[:], os: i64, bias: const f32[:], bo: i64, ro: i64, rows: i64, k: i64, mode: i64, sp: i64, tt: i64, pt: f32[:], pr: i64, hq: i32[:], hqs: f32[:]) {
    let nt: i64 = st[2];
    let nbn = (nt + 63) / 64;
    let cb = (cuda_block() >> tt) as! i32;
    let s4 = sp as! i32;
    let bm = (cb / s4) as i64;
    let ks = (cb % s4) as i64;
    let bn = cuda_block() & ((1 << tt) - 1);
    if bn >= nbn || bm * 128 >= rows { return; }
    let w = (cuda_thread() / 32) as! i32;
    let l = (cuda_thread() % 32) as! i32;
    let g = l / 4;
    let t = l % 4;
    let k4 = k / 4;
    let kb = k / 32;
    let r0 = bm * 128;
    let t0 = bn * 64;
    let sa = cuda_shared<i32>(4096);
    let sb = cuda_shared<i32>(2048);
    let sw = cuda_shared<f32>(512);
    let sx = cuda_shared<f32>(256);
    let sn = cuda_shared<f32>(512);
    var acc: f32[64] = [0.0; 64];
    let z: i32[4] = [0, 0, 0, 0];
    let np = kb / 2;
    let p0 = ((ks as! i32) * (np as! i32) / s4) as i64;
    let p1 = ((ks as! i32 + 1) * (np as! i32) / s4) as i64;
    if p0 < p1 { q8stage(wq, at, xq, xs, sa, sb, sw, sx, p0, 0, r0, t0, nt, k4, kb); }
    cuda_cp_commit();
    // ldmatrix: lane l gives row l % 8 of matrix l / 8.
    let mi = l / 8;
    let ri = l % 8;
    let xl = (ri / 2) % 4;
    let la = (w * 32 + (mi % 2) * 8 + ri) * 16;
    let lb = ((mi / 2) * 8 + ri) * 16;
    var s: i32 = 0;
    for p in p0..p1 {
        cuda_cp_wait(0);
        cuda_sync();
        if p + 1 < p1 { q8stage(wq, at, xq, xs, sa, sb, sw, sx, p + 1, 1 - s as i64, r0, t0, nt, k4, kb); }
        cuda_cp_commit();
        for jj in 2 {
            let ca = ((((jj as! i32) * 2 + mi / 2) ^ xl) * 4);
            let cb = ((((jj as! i32) * 2 + mi % 2) ^ xl) * 4);
            let fa = cuda_ldsm4(sa, s * 2048 + la + ca);
            let fc = cuda_ldsm4(sa, s * 2048 + la + 256 + ca);
            let wo = s * 256 + (jj as! i32) * 128 + w * 32 + g;
            let da = sw[wo];
            let db = sw[wo + 8];
            let dc = sw[wo + 16];
            let dd = sw[wo + 24];
            for np in 4 {
                let bb = cuda_ldsm4(sb, s * 1024 + (np as! i32) * 256 + lb + cb);
                for hf in 2 {
                    let n = np * 2 + hf;
                    let fb: i32[2] = [bb[hf * 2], bb[hf * 2 + 1]];
                    let c = cuda_mma(fa, fb, z);
                    let e = cuda_mma(fc, fb, z);
                    let xo = s * 128 + ((n as! i32) * 8 + t * 2) * 2 + (jj as! i32);
                    let s0 = sx[xo];
                    let s1 = sx[xo + 2];
                    acc[n * 8] = mul_add(c[0] as f32, da * s0, acc[n * 8]);
                    acc[n * 8 + 1] = mul_add(c[1] as f32, da * s1, acc[n * 8 + 1]);
                    acc[n * 8 + 2] = mul_add(c[2] as f32, db * s0, acc[n * 8 + 2]);
                    acc[n * 8 + 3] = mul_add(c[3] as f32, db * s1, acc[n * 8 + 3]);
                    acc[n * 8 + 4] = mul_add(e[0] as f32, dc * s0, acc[n * 8 + 4]);
                    acc[n * 8 + 5] = mul_add(e[1] as f32, dc * s1, acc[n * 8 + 5]);
                    acc[n * 8 + 6] = mul_add(e[2] as f32, dd * s0, acc[n * 8 + 6]);
                    acc[n * 8 + 7] = mul_add(e[3] as f32, dd * s1, acc[n * 8 + 7]);
                }
            }
        }
        s = 1 - s;
    }
    if pr > 0 {
        // silu(gate) * up an n-tile at a time, gate rows through sw and up rows
        // through sn; a token's 32 pairs one quantized block of h.
        const for <N> in [0, 1, 2, 3, 4, 5, 6, 7] {
            cuda_sync();
            const for <I> in [0, 1, 2, 3, 4, 5, 6, 7] {
                let lr = w * 32 + g + (I / 2) * 8;
                let tl = t * 2 + I % 2;
                if w < 2 { sw[tl * 64 + lr] = acc[N * 8 + I]; } else { sn[tl * 64 + lr - 64] = acc[N * 8 + I]; }
            }
            cuda_sync();
            for j in 2 {
                let tl = w * 2 + j as! i32;
                let tk = t0 + N * 8 + tl as i64;
                if tk < nt {
                    for q in 2 {
                        let gv = sw[tl * 64 + (q as! i32) * 32 + l];
                        wq32t(gv / (1 + exp(-gv)) * sn[tl * 64 + (q as! i32) * 32 + l], l as i64, tk, r0 / 64 + q, pr / 32, true, hq, hqs);
                    }
                }
            }
        }
        return;
    }
    let rr = r0 + (w * 32 + g) as i64;
    const for <N> in [0, 1, 2, 3, 4, 5, 6, 7] {
        const for <I> in [0, 1, 2, 3, 4, 5, 6, 7] {
            let r = rr + (I / 2) * 8;
            let tk = t0 + N * 8 + (t * 2) as i64 + I % 2;
            if r < rows && tk < nt {
                let v = acc[N * 8 + I];
                if sp > 1 { pt[(ks * 512 + tk) * rows + r] = v; }
                else {
                    let y = tk * os + ro + r;
                    if mode == 1 { o[y] += v; } else if mode == 2 { o[y] = v + bias[bo + ro + r]; } else { o[y] = v; }
                }
            }
        }
    }
}
// kmq8 for Q6_K's i8 copy: a scale a 16-block, so k16 mmas.
fn q6stage(wq: const i32[:], at: const f32[:], xq: const i32[:], xs: const f32[:], sa: i32[:], sb: i32[:], sw: f32[:], sx: f32[:], p: i64, s: i64, r0: i64, t0: i64, nt: i64, k4: i64, kb: i64) {
    let th = cuda_thread();
    for i in 4 {
        let e = th + i * 128;
        let r = e / 4;
        let c = e % 4;
        cuda_cp_async(sa, s * 2048 + r * 16 + ((c ^ ((r / 2) % 4)) * 4), wq, ((r0 / 128 * kb + p * 2 + c / 2) * 128 + r) * 8 + (c % 2) * 4);
    }
    for i in 2 {
        let e = th + i * 128;
        let tk = e / 4;
        let c = e % 4;
        cuda_cp_async(sb, s * 1024 + tk * 16 + ((c ^ ((tk / 2) % 4)) * 4), xq, min(t0 + tk, nt - 1) * k4 + p * 16 + c * 4);
    }
    cuda_cp_async(sw, s * 512 + th * 4, at, (r0 / 128 * (kb * 2) + p * 4) * 128 + th * 4);
    if th < 32 { cuda_cp_async(sx, s * 128 + th * 4, xs, p * 2048 + t0 * 2 + th * 4); }
}
cuda fn kmq6(wq: const i32[:], at: const f32[:], xq: const i32[:], xs: const f32[:], st: const i32[:], o: f32[:], os: i64, bias: const f32[:], bo: i64, ro: i64, rows: i64, k: i64, mode: i64, sp: i64, tt: i64, pt: f32[:], pr: i64, hq: i32[:], hqs: f32[:]) {
    let nt: i64 = st[2];
    let nbn = (nt + 63) / 64;
    let cb = (cuda_block() >> tt) as! i32;
    let s4 = sp as! i32;
    let bm = (cb / s4) as i64;
    let ks = (cb % s4) as i64;
    let bn = cuda_block() & ((1 << tt) - 1);
    if bn >= nbn || bm * 128 >= rows { return; }
    let w = (cuda_thread() / 32) as! i32;
    let l = (cuda_thread() % 32) as! i32;
    let g = l / 4;
    let t = l % 4;
    let k4 = k / 4;
    let kb = k / 32;
    let r0 = bm * 128;
    let t0 = bn * 64;
    let sa = cuda_shared<i32>(4096);
    let sb = cuda_shared<i32>(2048);
    let sw = cuda_shared<f32>(1024);
    let sx = cuda_shared<f32>(256);
    let sn = cuda_shared<f32>(512);
    var acc: f32[64] = [0.0; 64];
    let z: i32[4] = [0, 0, 0, 0];
    let np = kb / 2;
    let p0 = ((ks as! i32) * (np as! i32) / s4) as i64;
    let p1 = ((ks as! i32 + 1) * (np as! i32) / s4) as i64;
    if p0 < p1 { q6stage(wq, at, xq, xs, sa, sb, sw, sx, p0, 0, r0, t0, nt, k4, kb); }
    cuda_cp_commit();
    // ldmatrix: lane l gives row l % 8 of matrix l / 8.
    let mi = l / 8;
    let ri = l % 8;
    let xl = (ri / 2) % 4;
    let la = (w * 32 + mi * 8 + ri) * 16;
    let lb = (mi * 8 + ri) * 16;
    var s: i32 = 0;
    for p in p0..p1 {
        cuda_cp_wait(0);
        cuda_sync();
        if p + 1 < p1 { q6stage(wq, at, xq, xs, sa, sb, sw, sx, p + 1, 1 - s as i64, r0, t0, nt, k4, kb); }
        cuda_cp_commit();
        for jj in 2 {
            for hh in 2 {
                let ch = (((jj as! i32) * 2 + (hh as! i32)) ^ xl) * 4;
                let am = cuda_ldsm4(sa, s * 2048 + la + ch);
                let fa: i32[2] = [am[0], am[1]];
                let fc: i32[2] = [am[2], am[3]];
                let wo = s * 512 + ((jj as! i32) * 2 + (hh as! i32)) * 128 + w * 32 + g;
                let da = sw[wo];
                let db = sw[wo + 8];
                let dc = sw[wo + 16];
                let dd = sw[wo + 24];
                for nq in 2 {
                    let bm = cuda_ldsm4(sb, s * 1024 + (nq as! i32) * 512 + lb + ch);
                    for hf in 4 {
                        let n = nq * 4 + hf;
                        let fb = bm[hf];
                        let c = cuda_mma16(fa, fb, z);
                        let e = cuda_mma16(fc, fb, z);
                        let xo = s * 128 + ((n as! i32) * 8 + t * 2) * 2 + (jj as! i32);
                        let s0 = sx[xo];
                        let s1 = sx[xo + 2];
                        acc[n * 8] = mul_add(c[0] as f32, da * s0, acc[n * 8]);
                        acc[n * 8 + 1] = mul_add(c[1] as f32, da * s1, acc[n * 8 + 1]);
                        acc[n * 8 + 2] = mul_add(c[2] as f32, db * s0, acc[n * 8 + 2]);
                        acc[n * 8 + 3] = mul_add(c[3] as f32, db * s1, acc[n * 8 + 3]);
                        acc[n * 8 + 4] = mul_add(e[0] as f32, dc * s0, acc[n * 8 + 4]);
                        acc[n * 8 + 5] = mul_add(e[1] as f32, dc * s1, acc[n * 8 + 5]);
                        acc[n * 8 + 6] = mul_add(e[2] as f32, dd * s0, acc[n * 8 + 6]);
                        acc[n * 8 + 7] = mul_add(e[3] as f32, dd * s1, acc[n * 8 + 7]);
                    }
                }
            }
        }
        s = 1 - s;
    }
    if pr > 0 {
        // silu(gate) * up an n-tile at a time, gate rows through sw and up rows
        // through sn; a token's 32 pairs one quantized block of h.
        const for <N> in [0, 1, 2, 3, 4, 5, 6, 7] {
            cuda_sync();
            const for <I> in [0, 1, 2, 3, 4, 5, 6, 7] {
                let lr = w * 32 + g + (I / 2) * 8;
                let tl = t * 2 + I % 2;
                if w < 2 { sw[tl * 64 + lr] = acc[N * 8 + I]; } else { sn[tl * 64 + lr - 64] = acc[N * 8 + I]; }
            }
            cuda_sync();
            for j in 2 {
                let tl = w * 2 + j as! i32;
                let tk = t0 + N * 8 + tl as i64;
                if tk < nt {
                    for q in 2 {
                        let gv = sw[tl * 64 + (q as! i32) * 32 + l];
                        wq32t(gv / (1 + exp(-gv)) * sn[tl * 64 + (q as! i32) * 32 + l], l as i64, tk, r0 / 64 + q, pr / 32, true, hq, hqs);
                    }
                }
            }
        }
        return;
    }
    let rr = r0 + (w * 32 + g) as i64;
    const for <N> in [0, 1, 2, 3, 4, 5, 6, 7] {
        const for <I> in [0, 1, 2, 3, 4, 5, 6, 7] {
            let r = rr + (I / 2) * 8;
            let tk = t0 + N * 8 + (t * 2) as i64 + I % 2;
            if r < rows && tk < nt {
                let v = acc[N * 8 + I];
                if sp > 1 { pt[(ks * 512 + tk) * rows + r] = v; }
                else {
                    let y = tk * os + ro + r;
                    if mode == 1 { o[y] += v; } else if mode == 2 { o[y] = v + bias[bo + ro + r]; } else { o[y] = v; }
                }
            }
        }
    }
}
// The sp partial sums of split-K, in order, into o as mmb's mode says.
cuda fn kskr(pt: const f32[:], st: const i32[:], o: f32[:], os: i64, bias: const f32[:], bo: i64, ro: i64, rows: i64, mode: i64, sp: i64) {
    let i = cuda_block() * cuda_block_dim() + cuda_thread();
    let tk = i / rows;
    let r = i % rows;
    if tk >= st[2] { return; }
    var v: f32 = 0;
    for ks in sp { v += pt[(ks * 512 + tk) * rows + r]; }
    let y = tk * os + ro + r;
    if mode == 1 { o[y] += v; } else if mode == 2 { o[y] = v + bias[bo + ro + r]; } else { o[y] = v; }
}
// The batch's last token's quants, as the single-token kernels take them.
cuda fn klast(xq: const i32[:], xs: const f32[:], st: const i32[:], oq: i32[:], os: f32[:], om: f32[:], k: i64) {
    let i = cuda_block() * cuda_block_dim() + cuda_thread();
    let tk: i64 = st[2] - 1;
    if i < k / 4 { oq[i] = xq[tk * (k / 4) + i]; }
    if i < k / 32 { os[i] = xs[bsi(tk, i)]; om[i] = xs[bsi(tk, i) + 1024]; }
}
fn gmt(m: GM, xq: cuda::buf<i32>, xs: cuda::buf<f32>, o: cuda::buf<f32>, os: i64, bo: i64, ro: i64, mode: i64, mx: i64, pr: i64) {
    if m.rows == 0 { return; }
    if mx > 16 {
        // K split until there are about eight blocks an SM, for two token tiles.
        let rbk = (m.rows + 127) / 128;
        let sp = if pr > 0 { 1 } else { max(1, min(16, min(m.k / 256, 180 / (rbk * ((mx + 63) / 64))))) };
        let tt = if mx > 256 { 3 } else if mx > 128 { 2 } else if mx > 64 { 1 } else { 0 };
        let nb = (rbk * sp) << tt;
        if m.f == 1 { cuda_launch(kmq4, nb, 128, m.u, m.t, m.n, xq, xs, GBS, o, os, GV, bo, ro, m.rows, m.k, mode, sp, tt, GBSK, pr, GBH, GBHS); }
        else if m.f == 2 { cuda_launch(kmq6, nb, 128, m.u, m.t, xq, xs, GBS, o, os, GV, bo, ro, m.rows, m.k, mode, sp, tt, GBSK, pr, GBH, GBHS); }
        else { cuda_launch(kmq8, nb, 128, m.u, m.t, xq, xs, GBS, o, os, GV, bo, ro, m.rows, m.k, mode, sp, tt, GBSK, pr, GBH, GBHS); }
        if sp > 1 { cuda_launch(kskr, (mx * m.rows + 255) / 256, 256, GBSK, GBS, o, os, GV, bo, ro, m.rows, mode, sp); }
        return;
    }
    let nb = (m.rows + 3) / 4;
    if m.f == 1 { cuda_launch(kmt1, nb, 128, m.q, m.s, xq, xs, GBS, o, os, GV, bo, ro, m.rows, m.k, mode); }
    else if m.f == 2 { cuda_launch(kmt2, nb, 128, m.q, m.s, xq, xs, GBS, o, os, GV, bo, ro, m.rows, m.k, mode); }
    else { cuda_launch(kmt0, nb, 128, m.q, m.s, xq, xs, GBS, o, os, GV, bo, ro, m.rows, m.k, mode); }
}
// The batch in GBT at the position in GBS, up to mx tokens: with mx 16 every
// token's argmax to GBO, else the last token's logits to GLG and its argmax
// to GO.
fn gfwdt(mx: i64) {
    let d = C.d;
    let hd = C.hd;
    let kvd = C.nkv * hd;
    let qs = C.nh * hd + 2 * kvd;
    if GP { cuda::sync(); GC = clock(); }
    let e = GW[C.L * 5 + 1];
    cuda_launch(kembt, mx * ((d + 255) / 256), 256, e.f, e.q, e.s, GBX, GBT, GBS, d);
    gt(0);
    let s0: f32 = 1 / sqrt(hd as f32);
    for l in C.L {
        cuda_launch(knqt, mx, 512, GBX, GV, GVO[l * 3], GBQ, GBQS, GBS, d, C.eps, mx > 16);
        gt(1);
        let a = GW[l * 5];
        gmt(a, GBQ, GBQS, GBQKV, qs, GVO[l * 3 + 2], 0, 2, mx, 0);
        gt(2);
        gmt(GW[l * 5 + 1], GBQ, GBQS, GBQKV, qs, GVO[l * 3 + 2], a.rows, 2, mx, 0);
        gt(2);
        let nr = mx * (C.nh + 2 * C.nkv) * 64;
        cuda_launch(kropet, (nr + 127) / 128, 128, GBQKV, GKC, GVC, GKH, GVT, GRT, GBS, l * C.ctx, C.nh, C.nkv, C.neox, C.ctx, l * kvd * C.ctx);
        gt(3);
        if mx > 16 { let tt = if mx > 256 { 3 } else if mx > 128 { 2 } else if mx > 64 { 1 } else { 0 }; cuda_launch(kfa, C.nh << tt, 128, GBQKV, cuda::buf<i32> { p: GKH.p, n: GKH.n / 2 }, cuda::buf<i32> { p: GVT.p, n: GVT.n / 2 }, GBS, l * C.ctx * kvd, kvd, C.nh, C.nh / C.nkv, s0, tt, C.ctx, l * kvd * C.ctx, GBQ, GBQS); }
        else { cuda_launch(katt_t, mx * C.nh * 4, 128, GBQKV, GKC, GVC, GBPM, GBPA, GBS, l * C.ctx * kvd, kvd, C.nh, C.nh / C.nkv, s0, GBCNT, GBQ, GBQS); }
        gt(4);
        gmt(GW[l * 5 + 2], GBQ, GBQS, GBX, d, 0, 0, 1, mx, 0);
        gt(6);
        cuda_launch(knqt, mx, 512, GBX, GV, GVO[l * 3 + 1], GBQ, GBQS, GBS, d, C.eps, mx > 16);
        gt(1);
        if mx > 16 { gmt(GW[l * 5 + 3], GBQ, GBQS, GBGU, 2 * C.ff, 0, 0, 0, mx, C.ff); }
        else {
            gmt(GW[l * 5 + 3], GBQ, GBQS, GBGU, 2 * C.ff, 0, 0, 0, mx, 0);
            cuda_launch(kswt, mx * C.ff / 32 / 4, 128, GBGU, GBS, GBH, GBHS, C.ff);
        }
        gt(8);
        gt(7);
        gmt(GW[l * 5 + 4], GBH, GBHS, GBX, d, 0, 0, 1, mx, 0);
        gt(12);
    }
    cuda_launch(knqt, mx, 512, GBX, GV, GVO[C.L * 3], GBQ, GBQS, GBS, d, C.eps, mx > 16);
    gt(1);
    if mx > 16 {
        cuda_launch(klast, (d / 4 + 127) / 128, 128, GBQ, GBQS, GBS, GXQ, GXS, GXM, d);
        gt(11);
        gmv(GW[C.L * 5], GXQ, GXS, GXM, GLG, 0, 0, 0);
        gt(10);
        cuda_launch(kamax, 1, 1024, GLG, C.V, GO);
        gt(9);
        return;
    }
    gmt(GW[C.L * 5], GBQ, GBQS, GBLG, C.V, 0, 0, 0, mx, 0);
    gt(10);
    cuda_launch(kamaxt, mx, 1024, GBLG, C.V, GBS, GBO);
    gt(9);
}
// Tokens tk at positions pos.., at most 512, through a batch graph: up to 16
// get every argmax back, more the last token's.
fn gbatch(tk: const i32[:], pos: i64, out: i32[>..]&) {
    let big = tk.len > 16;
    var b: i32[>..] = [];
    b.resize(512, 0);
    for v, i in tk { b[i] = v; }
    cuda::upload(GBT, b);
    let s: i32[] = [0, pos as i32, tk.len as i32, 0];
    cuda::upload(GBS, s);
    // A graph a size: 16, then 64, 128, 256 and 512 for prompts.
    let gi = if !big { 0 } else if tk.len <= 64 { 1 } else if tk.len <= 128 { 2 } else if tk.len <= 256 { 3 } else { 4 };
    let mx = if gi == 0 { 16 } else { 32 << gi };
    if GP {
        gfwdt(mx);
    } else if GBP[gi] == 0 {
        gfwdt(mx);
        cuda::sync();
        cuda::capture_begin();
        gfwdt(mx);
        GBP[gi] = cuda::capture_end();
        cuda::upload(GBS, s);
    }
    if !GP { cuda::graph_run(GBP[gi]); }
    if big {
        var o: i32[] = [0, 0];
        cuda::download(GO, o);
        out.clear();
        out.push(o[0]);
        GBL = GLG.p;
        return;
    }
    out.resize(16, 0);
    cuda::download(GBO, out);
    out.resize(tk.len, 0);
    GBL = GBLG.p + ((tk.len - 1) * C.V * 4) as u64;
}
fn gballoc() {
    let d = C.d;
    let kvd = C.nkv * C.hd;
    let qs = C.nh * C.hd + 2 * kvd;
    let n = 512;
    GBT = cuda::alloc<i32>(n);
    GBS = cuda::alloc<i32>(4);
    GBO = cuda::alloc<i32>(16);
    GBX = cuda::alloc<f32>(n * d);
    GBQKV = cuda::alloc<f32>(n * qs);
    let mk = max(d, C.nh * C.hd);
    GBQ = cuda::alloc<i32>(n * mk / 4);
    GBQS = cuda::alloc<f32>((mk / 64 + 1) * 2048);
    GBGU = cuda::alloc<f32>(n * 2 * C.ff);
    GBH = cuda::alloc<i32>(n * C.ff / 4);
    GBHS = cuda::alloc<f32>((C.ff / 64 + 1) * 2048);
    GBPM = cuda::alloc<f32>(n * C.nh * 16 * 2);
    GBPA = cuda::alloc<f32>(n * C.nh * 16 * C.hd);
    GBLG = cuda::alloc<f32>(16 * C.V);
    GBSK = cuda::alloc<f32>(16 * n * 2048);
    GBCNT = cuda::alloc<i32>(n * C.nh);
    var z: i32[>..] = [];
    z.resize(n * C.nh, 0);
    cuda::upload(GBCNT, z);
}
// One forward of the token and position in GO, replayed from a graph once
// captured; `best` adds the argmax, which moves GO on to the next token.
fn gstep(best: bool) {
    if GP { gfwd(); if best { cuda_launch(kamax, 1, 1024, GLG, C.V, GO); } return; }
    if GG == 0 {
        var st: i32[] = [0, 0];
        cuda::download(GO, st);
        gfwd();
        cuda_launch(kamax, 1, 1024, GLG, C.V, GO);
        cuda::sync();
        cuda::capture_begin();
        gfwd();
        GG = cuda::capture_end();
        cuda::capture_begin();
        gfwd();
        cuda_launch(kamax, 1, 1024, GLG, C.V, GO);
        GA = cuda::capture_end();
        cuda::upload(GO, st);
    }
    cuda::graph_run(if best { GA } else { GG });
}
fn gset(tok: i64, pos: i64) {
    let st: i32[] = [tok as i32, pos as i32];
    cuda::upload(GO, st);
}
fn gnext(temp: f32, topk: i64, topp: f32, r: rng&) -> i64 {
    var st: i32[] = [0, 0];
    if temp <= 0 {
        cuda::download(GO, st);
        return st[0];
    }
    var lg: f32[>..] = [];
    lg.resize(C.V, 0);
    cuda::download(GLG, lg);
    CD.clear();
    for v, i in lg { CD.push(v); CD.push(i as f32); }
    CK = C.V;
    return sample(0, temp, topk, topp, r);
}
fn ggen(toks: const i32[:], n: i64, temp: f32, topk: i64, topp: f32, r: rng&, t1: f64) {
    for i in 16 { GT[i] = 0; }
    let nw = C.nw;
    C.nw = 1;
    let gr = temp <= 0;
    var y: i32[>..] = [];
    var c0 = 0;
    while c0 < toks.len { let c1 = min(c0 + 512, toks.len); gbatch(toks[c0..c1], c0, y); c0 = c1; }
    var nt: i64 = y[y.len - 1];
    if !gr {
        var lg: f32[>..] = [];
        lg.resize(C.V, 0);
        cuda_copy_out(lg[0..C.V], GBL);
        CD.clear();
        for v, i in lg { CD.push(v); CD.push(i as f32); }
        CK = C.V;
        nt = sample(0, temp, topk, topp, r);
    }
    let t2 = clock();
    var pos = toks.len;
    var hist: i32[>..] = [];
    hist.append(toks);
    var out: u8[>..] = [];
    var dr: i32[>..] = [];
    var bt: i32[>..] = [];
    var g = 0;
    var passes = 0;
    var acc = 0;
    while vt[nt] != 3 && g < n && pos < C.ctx - 1 {
        emit(nt, out);
        hist.push(nt as i32);
        g++;
        if g >= n { break; }
        dr.clear();
        if gr && DK > 0 { draft(hist, min(min(DK, 15), min(n - g, C.ctx - pos - 2)), dr); }
        if dr.len == 0 {
            gset(nt, pos);
            gstep(gr);
            pos++;
            nt = gnext(temp, topk, topp, r);
            continue;
        }
        bt.clear();
        bt.push(nt as i32);
        bt.append(dr);
        gbatch(bt, pos, y);
        passes++;
        var a = 0;
        while a < dr.len && y[a] == dr[a] { a++; }
        acc += a;
        var stop = false;
        for i in a {
            let tk = dr[i];
            if vt[tk] == 3 || g >= n { stop = true; break; }
            emit(tk, out);
            hist.push(tk);
            g++;
        }
        if stop { break; }
        pos += 1 + a;
        nt = y[a];
    }
    let t3 = clock();
    C.nw = nw;
    if GP {
        let nm = ["emb", "norm", "qkv", "rope", "attn", "quant", "o", "swiglu", "gate_up", "-", "lm_head", "?", "down"];
        var tot: f64 = 0;
        for v in GT { tot += v; }
        for k in 13 { if GT[k] > 0 { print(nm[k], " ", (GT[k] * 1000) as i64, " ms ", (GT[k] * 100 / tot) as i64, "%"); } }
    }
    var st: u8[>..] = [];
    format(st, "\n\nprompt ", toks.len, " tok ");
    format_flt(st, toks.len / (t2 - t1), 1);
    format(st, " tok/s | gen ", g, " tok ");
    format_flt(st, g / (t3 - t2), 2);
    format(st, " (", ((t3 - t2) * 1000) as i64, " ms) tok/s\n");
    write_stderr(st);
}

fn main() {
    let a = args();
    var model: u8[>..] = [];
    var prompt: u8[>..] = [];
    var n = 64;
    var nw = 0;
    var ctx = 2048;
    var temp: f32 = 0;
    var topk = 40;
    var topp: f32 = 0.95;
    var seed = 42;
    var chat = false;
    var verb = false;
    var pf: u8[>..] = [];
    var i = 1;
    while i < a.len {
        let k = a[i];
        let v = if i + 1 < a.len { a[i + 1] } else { a[i] };
        if k == "-m" { model.append(v); i++; }
        else if k == "-p" { prompt.append(v); i++; }
        else if k == "-f" { guard read_file(v, prompt) else { abort(str("cannot read ", v)); } i++; }
        else if k == "-n" { n = parse_int(v); i++; }
        else if k == "-t" { nw = parse_int(v); i++; }
        else if k == "-c" { ctx = parse_int(v); i++; }
        else if k == "-b" { NB = parse_int(v); i++; }
        else if k == "--spin" { SP = parse_int(v); i++; }
        else if k == "--fbatch" { FB = parse_int(v) != 0; i++; }
        else if k == "--pin" { PIN = parse_int(v) != 0; i++; }
        else if k == "--pfb" { PFB = parse_int(v); i++; }
        else if k == "--draft" { DK = parse_int(v); i++; }
        else if k == "--tree" { MB = max(parse_int(v), 1); i++; }
        else if k == "--tree-nodes" { TB = max(parse_int(v), 2); i++; }
        else if k == "--temp" { temp = parse_flt(v) as f32; i++; }
        else if k == "--top-k" { topk = parse_int(v); i++; }
        else if k == "--top-p" { topp = parse_flt(v) as f32; i++; }
        else if k == "--seed" { seed = parse_int(v); i++; }
        else if k == "--chat" { chat = true; }
        else if k == "-v" { verb = true; }
        else if k == "--ppl" { pf.append(v); i++; }
        else if k == "--gpu" { GPU = true; }
        else if k == "--gprof" { GPU = true; GP = true; }
        i++;
    }
    guard model.len > 0 else { print("usage: llama -m model.gguf (-p prompt | -f file | --ppl file) [-n 64] [-t 8] [-c 2048] [-b 256] [--temp 0] [--top-k 40] [--top-p 0.95] [--seed 42] [--chat] [-v]"); return; }
    topk = if temp <= 0 { 1 } else { min(max(topk, 1), 1000) };
    nw = min(max(nw, 0), 16);
    queue_spin(SP);
    let t0 = clock();
    var ids: i64[>..] = [];
    load(model, nw, ctx, ids);
    if PIN { pin_thread(2 * C.nw); }
    let t1 = clock();
    if pf.len > 0 { ppl(pf); if !GPU { quit(verb); } for id in ids { thread_wait(id); } return; }
    var text: u8[>..] = [];
    if chat { format(text, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n", prompt, "<|im_end|>\n<|im_start|>assistant\n"); }
    else { text.append(prompt); }
    var toks: i32[>..] = [];
    let te0 = clock();
    encode(text, toks);
    if verb { print("encode ms ", (clock() - te0) * 1000); }
    if toks.len == 0 { toks.push(151643); }
    if verb { print("tokens: ", toks); }
    var r = rng { seed as u64 };
    var g = 0;
    if GPU { if n > 0 { ggen(toks, n, temp, topk, topp, r, t1); } return; }
    if n > 0 {
        prefill(toks, 0, topk);
        let t2 = clock();
        var pos = toks.len;
        var out: u8[>..] = [];
        var hist: i32[>..] = [];
        hist.append(toks);
        var nt = sample(0, temp, topk, topp, r);
        var kk = min(DK, 3);
        var cool = 0;
        var miss = 0;
        var ema: f64 = 1.0;
        var passes = 0;
        var drafted = 0;
        var acc = 0;
        loop {
            if vt[nt] == 3 || g >= n || pos >= C.ctx - 1 { break; }
            emit(nt, out);
            hist.push(nt as i32);
            g++;
            if g >= n { break; }
            var bt: i32[>..] = [];
            bt.push(nt as i32);
            TP.clear();
            TD.clear();
            TP.push(-1);
            TD.push(0);
            let room = min(n - g, C.ctx - pos - 2);
            if temp <= 0 && cool == 0 && room > 0 { dtree(hist, min(kk, room), bt); } else if cool > 0 { cool--; }
            fwd(bt, pos, 1, 0, topk);
            passes++;
            let nd = bt.len - 1;
            drafted += nd;
            var path: i64[>..<] = [];
            var cur = 0;
            loop {
                let y = sample(cur, temp, topk, topp, r);
                var nx = -1;
                for c in 1..bt.len { if TP[c] == cur && bt[c] == y { nx = c; break; } }
                guard nx >= 0 && path.len < room else { break; }
                path.push(nx);
                cur = nx;
            }
            let a = path.len;
            acc += a;
            if nd > 0 {
                kk = if a >= min(kk, nd) && DK >= 7 { 7 } else { min(DK, 3) };
                ema = 0.75 * ema + 0.25 * a / min(nd, kk);
                if a == 0 { miss++; cool = min(1 << miss, 16); } else { miss = 0; }
                if ema < 0.35 { cool = max(cool, 32); ema = 0.6; }
            }
            var stop = false;
            for c, i in path {
                let tk = bt[c];
                if vt[tk] == 3 || g >= n { stop = true; break; }
                emit(tk, out);
                hist.push(tk);
                g++;
                if c != i + 1 { KCP.push(pos + c); KCP.push(pos + i + 1); }
            }
            if stop { break; }
            pos += 1 + a;
            nt = sample(cur, temp, topk, topp, r);
            TP.clear();
            TD.clear();
        }
        if verb { print("\nspec: ", passes, " passes, ", drafted, " drafted, ", acc, " accepted"); }
        let t3 = clock();
        var st: u8[>..] = [];
        format(st, "\n\nload ");
        format_flt(st, t1 - t0, 2);
        format(st, " s | prompt ", toks.len, " tok ");
        format_flt(st, toks.len / (t2 - t1), 1);
        format(st, " tok/s | gen ", g, " tok ");
        format_flt(st, g / (t3 - t2), 2);
        format(st, " (", ((t3 - t2) * 1000) as i64, " ms)");
        format(st, " tok/s\n");
        write_stderr(st);
    }
    quit(verb);
    for id in ids { thread_wait(id); }
}
