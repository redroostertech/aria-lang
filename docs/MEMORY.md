# Aria Memory Model — Design Decision

> Status: **decided** (research-backed), not yet implemented.
> Basis: a deep multi-source review (20 sources, 25 claims adversarially
> verified; 20 confirmed, 5 refuted). Citations inline.

## Decision

Aria uses **optimized reference counting with compile-time reuse analysis**
(the Perceus / Koka lineage) as its default, zero-annotation automatic memory
mechanism — *not* region inference, and *not* tracing GC. Immutability-by-default
is the amplifier that makes this fast. Static lifetime analysis is demoted to an
opportunistic fast-path, and a rare cycle fallback covers the residual case.

The programmer writes **no memory annotations**. Performance targets
manual/ownership-class behavior, with deterministic destruction (no GC pauses).

## The tiered model

```
CORE (default, everywhere)
  Perceus-style optimized RC + reuse analysis (FBIP, borrow inference)
  → zero annotations, deterministic, no pauses, mutates in-place when unique

AMPLIFIER
  Immutability-by-default
  → cycles rare-by-construction · maximal in-place reuse · provable-immutable ⇒ elide checks

STATIC FAST-PATH
  Escape analysis + bounded local/stack allocation  (NOT whole-program regions)

DETERMINISTIC CLEANUP (files / sockets / handles)
  Linear/affine handle types (RAII-like), not finalizers

RARE FALLBACK
  Cycle collector — or concurrent tracing only for cycles (LXR-style) —
  and/or a WasmGC backend on the web

FRONTIER (research, opt-in later — do NOT bet the core on it)
  Ownership/borrow inference + generational references to shave residual RC cost
```

## Evidence summary

| Approach | Verdict | Confidence | Source |
|---|---|---|---|
| Optimized RC + reuse (Perceus) | ✅ The core — matches/beats manual, zero annotations | high (3-0) | [MS Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/11/flreuse-tr.pdf), [Perceus](https://xnning.github.io/papers/perceus.pdf) |
| Immutability amplifies RC | ✅ Cycles rare, more reuse, check elision | high (3-0) | MS Research, [Vale](https://verdagon.dev/blog/zero-cost-borrowing-regions-overview) |
| Pure region inference as sole scheme | ❌ Over-retains, space leaks linear in runtime, brittle | high (3-0) | [MLKit](https://link.springer.com/article/10.1023/B:LISP.0000029446.78563.a4), [Aiken PLDI'95](https://theory.stanford.edu/~aiken/publications/papers/pldi95.pdf) |
| Escape analysis / bounded stack alloc | ✅ Good as a fast-path layer | high | Aiken PLDI'95 |
| RC-centric + rare cycle trace (LXR) | ✅ Beats top tracing GC on throughput *and* tail latency | high (3-0) | [arXiv 2210.17175](https://arxiv.org/abs/2210.17175) |
| Generational references (Vale) | ⚠️ Promising (~10.8% overhead), self-reported, unproven | medium (3-0) | [verdagon.dev](https://verdagon.dev/blog/generational-references) |
| ASAP (fully static frees) | ⚠️ Coherent design, no performance proof | medium (3-0) | [Cambridge TR-908](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-908.html) |
| Cyclone (inferred regions) | annotation burden ~0.5%, but up to ~3× compute overhead | high (3-0) | [Cyclone](https://www.cs.umd.edu/projects/cyclone/papers/cyclone-regions.pdf) |
| WasmGC backend | ✅ Smaller modules (2.3 vs 6–9.6 KB) but ties to host-GC semantics | high (3-0) | [v8.dev](https://v8.dev/blog/wasm-gc-porting) |

### Key caveat on the headline number
Koka's ~19–30% win over C++ `std::map` is real and primary-sourced, but the
authors disclose it shrinks to **parity on ARM64/M1 with clang** and is partly a
TRMC/allocator-alignment artifact. Read it as "can match or beat manual on
specific benchmarks," not a universal guarantee.

## Claims that were refuted (do not over-claim)

- ❌ "Region inference + GC is as efficient as a state-of-the-art generational GC" (1-2)
- ❌ "Vale: 2–10% overhead vs Java 89% / Go 183% / Swift 320%" (0-3) — marketing framing
- ❌ "Pure functions make safety zero-cost" (1-2)
- ❌ "Cyclone is fully static with no runtime checks" (0-3) — it has runtime checks
- ❌ "Generation checks are the sole source of Vale's overhead" (0-3)

## Open questions / risks

1. **No primary source benchmarks the full integrated blend end-to-end.**
   Per-component evidence is strong; the combination is unmeasured. Real risk.
2. **Can ownership be *inferred* (not annotated) at whole-program scale?**
   This is Aria's key bet. Our AI-native edge — we control the canonical code
   forms the model emits — is exactly what might make inference tractable where
   it failed for human-oriented languages. Unproven, high upside.
3. **Deterministic cleanup without annotations** — linear/affine handle types
   look right, but can linearity be inferred rather than declared?
4. **WasmGC vs. compile-our-own-RC-to-WASM-MVP** — code-size favors WasmGC;
   portability/uniform semantics favor RC. Likely: RC core + optional WasmGC backend.

## What prior languages prove each component is viable

- **Optimized RC + reuse**: Koka, Roc (ship it; competitive performance).
- **Rare cycle trace alongside RC**: LXR (beats Shenandoah on throughput + tail latency).
- **Low annotation burden via inference**: Cyclone (~0.5% of code).
- **Safety without GC or borrow checker**: Vale (generational references).
- **Smaller web modules**: WasmGC (host-managed memory).
