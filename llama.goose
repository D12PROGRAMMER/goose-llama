import std;
import os;
import math;
import binary;
import dictionary;

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
            let b1 = min(b0 + NB, n);
            let a0 = max(b0, h);
            if b1 > h { fwd(tk[st + b0..st + b1], b0, 2, a0 - b0, 0); } else { fwd(tk[st + b0..st + b1], b0, 0, 0, 0); }
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
    if pf.len > 0 { ppl(pf); quit(verb); for id in ids { thread_wait(id); } return; }
    var text: u8[>..] = [];
    if chat { format(text, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n", prompt, "<|im_end|>\n<|im_start|>assistant\n"); }
    else { text.append(prompt); }
    var toks: i32[>..] = [];
    encode(text, toks);
    if toks.len == 0 { toks.push(151643); }
    if verb { print("tokens: ", toks); }
    var r = rng { seed as u64 };
    var g = 0;
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
