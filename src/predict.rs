//! Byte-level predictive model for neural-style lossless compression.
//!
//! This is the "front end" that produces probabilities; a coder such as the
//! arithmetic/rANS back end turns those probabilities into bits. The model is
//! adaptive: it learns the structure of the stream as it sees it, with no
//! up-front statistics pass.
//!
//! The design mirrors the classic context-mixing (PAQ-style) compressors:
//!
//!   * Each byte is predicted one bit at a time, MSB first.
//!   * Several *context models* each guess the probability that the next bit is
//!     a 1, conditioned on a hashed context (the bytes seen so far + the bits
//!     already decoded inside the current byte).
//!   * The individual predictions are combined with a logistic mixer whose
//!     weights are trained online by gradient descent.
//!
//! We implement order-0, order-1 and order-2 context models. Everything uses
//! 12-bit fixed-point probabilities (1..=4095) so it lines up with the 12-bit
//! `SCALE_BITS` used by the rANS coder elsewhere in Aria.

// Standard library only; no external crates.

/// Number of bits of fixed-point precision for a probability. A probability is
/// stored as an integer in `1..=4095`, i.e. `p / 4096`.
const PROB_BITS: u32 = 12;
const PROB_ONE: u32 = 1 << PROB_BITS; // 4096
const PROB_MAX: u16 = (PROB_ONE - 1) as u16; // 4095
const PROB_MIN: u16 = 1;

/// Size (in entries) of each hashed context table. A power of two so we can
/// mask instead of dividing. 1<<22 = ~4M entries keeps collisions rare while
/// staying memory-friendly.
const TABLE_BITS: u32 = 22;
const TABLE_SIZE: usize = 1 << TABLE_BITS;
const TABLE_MASK: u32 = (TABLE_SIZE as u32) - 1;

/// Number of context models we mix together (order 0, 1, 2).
const NUM_MODELS: usize = 3;

// ---------------------------------------------------------------------------
// Stretch / squash: the logistic domain transforms used by the mixer.
//
// `squash(x)` maps a real "stretched" value back to a 12-bit probability via
// the logistic function. `stretch(p)` is its inverse (logit). They are kept as
// integer tables for speed and determinism. We compute them with f64 at table
// build time; the *runtime* path is fully integer and reproducible.
// ---------------------------------------------------------------------------

struct Logistic {
    // stretch[p] for p in 0..4096, clamped logit scaled by 256.
    stretch: Vec<i32>,
    // squash[x + 2047] for x in -2047..=2047: the 12-bit logistic of a scaled
    // logit. A precomputed table (NOT live f64) so the predict path is fully
    // integer and bit-identical across platforms/libm — required so a stream
    // compressed on one machine decompresses losslessly on another.
    squash: Vec<u16>,
}

impl Logistic {
    fn new() -> Self {
        let mut stretch = vec![0i32; PROB_ONE as usize];
        for (p, slot) in stretch.iter_mut().enumerate() {
            // Avoid the singularities at 0 and 1.
            let pf = (p as f64 + 0.5) / PROB_ONE as f64;
            let logit = (pf / (1.0 - pf)).ln();
            // Scale by 256 and clamp to a sane range.
            let v = (logit * 256.0).round() as i32;
            *slot = v.clamp(-2047, 2047);
        }
        // Precompute the squash table over the clamped logit domain so the
        // runtime path performs no f64 arithmetic.
        let mut squash = vec![0u16; 4095];
        for (idx, slot) in squash.iter_mut().enumerate() {
            let x = idx as i32 - 2047;
            let xf = x as f64 / 256.0;
            let p = 1.0 / (1.0 + (-xf).exp());
            let pi = (p * PROB_ONE as f64).round() as i32;
            *slot = pi.clamp(PROB_MIN as i32, PROB_MAX as i32) as u16;
        }
        Logistic { stretch, squash }
    }

    #[inline]
    fn stretch(&self, p: u16) -> i32 {
        self.stretch[p as usize]
    }

    /// Inverse of `stretch`: map a scaled logit back to a 12-bit probability via
    /// a precomputed integer table (no runtime float), for cross-platform
    /// determinism. `x` is clamped to the table's [-2047, 2047] domain.
    #[inline]
    fn squash(&self, x: i32) -> u16 {
        let x = x.clamp(-2047, 2047);
        self.squash[(x + 2047) as usize]
    }
}

// ---------------------------------------------------------------------------
// A single adaptive bit counter stored as a 12-bit probability. On update it
// moves toward the observed bit by a fixed fraction (an exponential moving
// average). This is the per-context state.
// ---------------------------------------------------------------------------

#[inline]
fn counter_update(p: &mut u16, bit: u8, rate: u32) {
    // p += (target - p) >> rate, with target = 0 or 4095.
    let cur = *p as i32;
    let target = if bit == 1 { PROB_MAX as i32 } else { 0 };
    let next = cur + ((target - cur) >> rate);
    *p = next.clamp(PROB_MIN as i32, PROB_MAX as i32) as u16;
}

// ---------------------------------------------------------------------------
// The predictor.
// ---------------------------------------------------------------------------

/// Adaptive context-mixing predictor. Predicts a byte one bit at a time.
pub struct Predictor {
    log: Logistic,

    /// One probability table per context model. Each table maps a hashed
    /// context to a 12-bit bit probability, initialised to 1/2 (= 2048).
    tables: Vec<Vec<u16>>,

    /// Mixer weights (one per model), fixed-point scaled by 65536.
    weights: [i32; NUM_MODELS],

    /// Byte history needed to form contexts.
    c1: u32, // most recent byte
    c2: u32, // byte before that

    /// State for the byte currently being predicted.
    /// `partial` holds the bits seen so far this byte, with a leading 1 sentinel
    /// (so it doubles as a "how many bits so far" indicator). Range 1..=255 plus
    /// the final 256..=511 before reset.
    partial: u32,

    /// Cached per-bit working state filled in by `predict_bit` and consumed by
    /// `update_bit`: the table indices used and the individual stretched preds.
    idx: [usize; NUM_MODELS],
    st: [i32; NUM_MODELS],
    last_p: u16,
}

impl Predictor {
    /// Create a fresh predictor with all contexts at probability 1/2.
    pub fn new() -> Self {
        let tables = (0..NUM_MODELS)
            .map(|_| vec![(PROB_ONE / 2) as u16; TABLE_SIZE])
            .collect();
        Predictor {
            log: Logistic::new(),
            tables,
            weights: [(1 << 16) / NUM_MODELS as i32; NUM_MODELS],
            c1: 0,
            c2: 0,
            partial: 1,
            idx: [0; NUM_MODELS],
            st: [0; NUM_MODELS],
            last_p: (PROB_ONE / 2) as u16,
        }
    }

    /// Hash a model order's context together with the partial byte into a table
    /// index. `order` selects how much history is folded in.
    #[inline]
    fn ctx_index(&self, order: usize) -> usize {
        // Mix the relevant history bytes and the partial byte with a couple of
        // multiply-xor rounds (a small, deterministic hash).
        let mut h: u32 = self.partial.wrapping_mul(0x9E37_79B1);
        h ^= (order as u32 + 1).wrapping_mul(0x85EB_CA77);
        if order >= 1 {
            h = h.wrapping_add(self.c1).wrapping_mul(0xC2B2_AE3D);
            h ^= h >> 15;
        }
        if order >= 2 {
            h = h.wrapping_add(self.c2).wrapping_mul(0x27D4_EB2F);
            h ^= h >> 13;
        }
        h ^= h >> 16;
        (h & TABLE_MASK) as usize
    }

    /// Predict the probability that the next bit is 1, as a 12-bit value in
    /// `1..=4095`. Does not mutate learned state, but caches working values for
    /// the subsequent `update_bit` call.
    pub fn predict_bit(&mut self) -> u16 {
        let mut dot: i64 = 0;
        for m in 0..NUM_MODELS {
            let i = self.ctx_index(m);
            let p = self.tables[m][i];
            let s = self.log.stretch(p);
            self.idx[m] = i;
            self.st[m] = s;
            dot += (self.weights[m] as i64) * (s as i64);
        }
        // weights are scaled by 65536.
        let mixed = (dot >> 16) as i32;
        let p = self.log.squash(mixed);
        self.last_p = p;
        p
    }

    /// Feed the actual bit back in, training both the context tables and the
    /// mixer weights. Must be called once after each `predict_bit`.
    pub fn update_bit(&mut self, bit: u8) {
        let bit = bit & 1;

        // --- Train the mixer (online logistic gradient). ---
        // error = actual - predicted, in probability units.
        let err = (bit as i32) * (PROB_ONE as i32) - self.last_p as i32; // ~ -4095..4095
        // learning rate: small shift keeps things stable.
        const LR_SHIFT: i32 = 10;
        for m in 0..NUM_MODELS {
            let dw = (self.st[m] * err) >> LR_SHIFT;
            // Clamp (PAQ-style) so the accumulator can never overflow i32 on a
            // long adversarial stream, and stays numerically stable.
            self.weights[m] = (self.weights[m] + (dw >> 6)).clamp(-(1 << 20), 1 << 20);
        }

        // --- Train each context counter toward the observed bit. ---
        // Faster adaptation for higher orders (they see less data per context).
        for m in 0..NUM_MODELS {
            let i = self.idx[m];
            let rate = match m {
                0 => 5, // order-0: slow, stable
                1 => 4,
                _ => 3, // order-2: fast
            };
            let mut p = self.tables[m][i];
            counter_update(&mut p, bit, rate);
            self.tables[m][i] = p;
        }

        // --- Advance the partial-byte / history state. ---
        self.partial = (self.partial << 1) | bit as u32;
        if self.partial >= 256 {
            // Completed a byte. The low 8 bits are the byte value.
            let byte = self.partial & 0xFF;
            self.c2 = self.c1;
            self.c1 = byte;
            self.partial = 1;
        }
    }

    /// Convenience helper: predict the full distribution over the 8 bits of the
    /// next byte without committing to any path. Returns the 8 per-bit
    /// probabilities (P(bit==1)) for the *most likely* MSB-first path, i.e. it
    /// greedily walks the bit tree. Useful for diagnostics / debugging.
    ///
    /// This temporarily mutates and then restores the partial-byte state so the
    /// caller's encoding position is unchanged.
    pub fn next_byte_probs(&mut self) -> [u16; 8] {
        let saved_partial = self.partial;
        let saved_idx = self.idx;
        let saved_st = self.st;
        let saved_last = self.last_p;

        let mut out = [0u16; 8];
        for slot in out.iter_mut() {
            let p = self.predict_bit();
            *slot = p;
            // Walk toward the more likely bit without learning.
            let bit = if p >= (PROB_ONE / 2) as u16 { 1 } else { 0 };
            self.partial = (self.partial << 1) | bit;
            if self.partial >= 256 {
                self.partial = 1;
            }
        }

        // Restore: this helper must not change learned or positional state.
        self.partial = saved_partial;
        self.idx = saved_idx;
        self.st = saved_st;
        self.last_p = saved_last;
        out
    }
}

impl Default for Predictor {
    fn default() -> Self {
        Predictor::new()
    }
}

/// Run the predictor over `data` and return the average cross-entropy in
/// **bits per byte**: `sum over bits of -log2(p_correct) / num_bytes`.
///
/// A value near 8.0 means the model learned nothing (random); values well
/// below 8.0 mean it is finding structure it could exploit for compression.
pub fn eval_bits_per_byte(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut pred = Predictor::new();
    let mut total_bits = 0.0f64;

    for &byte in data {
        for k in (0..8).rev() {
            let bit = (byte >> k) & 1;
            let p1 = pred.predict_bit(); // P(bit == 1), 1..=4095
            let p_one = p1 as f64 / PROB_ONE as f64;
            let p_correct = if bit == 1 { p_one } else { 1.0 - p_one };
            // p_correct is always in (0,1) thanks to the clamp, so log2 is safe.
            total_bits += -p_correct.log2();
            pred.update_bit(bit);
        }
    }

    total_bits / data.len() as f64
}

/// Print a small demonstration: bits/byte for a highly structured sample versus
/// random data. The structured sample should compress far below 8 bits/byte;
/// the random sample should sit near 8.
pub fn demo() {
    // Highly repetitive text.
    let mut repetitive = Vec::new();
    for _ in 0..400 {
        repetitive.extend_from_slice(b"the quick brown fox ");
    }

    // The alphabet, repeated.
    let mut alphabet = Vec::new();
    for _ in 0..400 {
        alphabet.extend_from_slice(b"abcdefghijklmnopqrstuvwxyz");
    }

    // Pseudo-random bytes from a simple LCG (deterministic, no crates).
    let random = lcg_bytes(8000, 0x1234_5678_9ABC_DEF0);

    println!("Aria predictive model — bits/byte (lower = more structure found)");
    println!("  repetitive text : {:.3} bits/byte", eval_bits_per_byte(&repetitive));
    println!("  repeated alphabet: {:.3} bits/byte", eval_bits_per_byte(&alphabet));
    println!("  random bytes     : {:.3} bits/byte", eval_bits_per_byte(&random));
}

/// Deterministic pseudo-random byte stream from a 64-bit LCG. Used only for the
/// demo and tests so we stay crate-free and reproducible.
fn lcg_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // Numerical Recipes LCG constants.
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Use a high byte to avoid the low-bit periodicity of an LCG.
        out.push((state >> 56) as u8);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinism_same_input_same_result() {
        let data = b"the quick brown fox jumps over the lazy dog, repeatedly!";
        let mut blob = Vec::new();
        for _ in 0..50 {
            blob.extend_from_slice(data);
        }
        let a = eval_bits_per_byte(&blob);
        let b = eval_bits_per_byte(&blob);
        assert_eq!(a.to_bits(), b.to_bits(), "model must be deterministic");
    }

    #[test]
    fn repetitive_text_well_below_8() {
        let mut blob = Vec::new();
        for _ in 0..400 {
            blob.extend_from_slice(b"the quick brown fox ");
        }
        let bpb = eval_bits_per_byte(&blob);
        assert!(bpb < 2.0, "repetitive text should be < 2.0 bits/byte, got {bpb}");
    }

    #[test]
    fn alphabet_compresses_strongly() {
        let mut blob = Vec::new();
        for _ in 0..400 {
            blob.extend_from_slice(b"abcdefghijklmnopqrstuvwxyz");
        }
        let bpb = eval_bits_per_byte(&blob);
        // 26 equiprobable symbols would be log2(26) ~= 4.7 if memoryless, but the
        // fixed cycle is fully predictable, so we expect far less.
        assert!(bpb < 2.0, "repeated alphabet should compress strongly, got {bpb}");
    }

    #[test]
    fn random_is_near_eight() {
        let data = lcg_bytes(8000, 0xDEAD_BEEF_CAFE_F00D);
        let bpb = eval_bits_per_byte(&data);
        // Random data carries ~8 bits of entropy per byte; the adaptive model
        // can't beat it and should hover around 8 (allow some slack).
        assert!(bpb > 7.0, "random data should be near 8 bits/byte, got {bpb}");
        assert!(bpb < 8.6, "model should not badly inflate random data, got {bpb}");
    }

    #[test]
    fn empty_input_is_zero() {
        assert_eq!(eval_bits_per_byte(&[]), 0.0);
    }

    #[test]
    fn probabilities_in_range() {
        let mut pred = Predictor::new();
        for byte in 0u16..512 {
            let b = (byte & 0xFF) as u8;
            for k in (0..8).rev() {
                let p = pred.predict_bit();
                assert!((PROB_MIN..=PROB_MAX).contains(&p), "p out of range: {p}");
                pred.update_bit((b >> k) & 1);
            }
        }
    }

    #[test]
    fn next_byte_probs_does_not_disturb_state() {
        let mut pred = Predictor::new();
        // Train a little.
        for &byte in b"hello hello hello hello" {
            for k in (0..8).rev() {
                let _ = pred.predict_bit();
                pred.update_bit((byte >> k) & 1);
            }
        }
        let p_before = pred.predict_bit();
        let _probs = pred.next_byte_probs();
        let p_after = pred.predict_bit();
        assert_eq!(p_before, p_after, "next_byte_probs must not change state");
    }
}
