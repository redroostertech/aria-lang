//! Binary arithmetic (range) coder — the entropy back end for predictive
//! (neural) compression in Aria.
//!
//! Where `rans` codes against a *static* order-0 byte histogram, this module
//! codes one bit at a time against an *externally supplied* probability. That
//! is exactly the interface a predictive model wants: a network (or any
//! context model) emits "the next bit is 1 with probability p", the coder
//! turns that prediction into a fraction of a bit, and the decoder — given the
//! identical prediction — recovers the bit. Because the model is driven by
//! already-decoded data, encoder and decoder stay in lockstep with no side
//! table to transmit.
//!
//! The implementation is the classic 32-bit binary range coder used by LZMA,
//! with explicit carry handling via a one-byte cache plus a run-length of
//! pending `0xFF` bytes. Probabilities are 12-bit (`1..=4095`, i.e. p(bit==1)
//! scaled by 4096). Correctness / exact losslessness is the absolute priority;
//! every public path is exercised by the round-trip tests at the bottom.

const PROB_BITS: u32 = 12;
const PROB_SCALE: u32 = 1 << PROB_BITS; // 4096
const TOP: u32 = 1 << 24; // renormalize when range drops below this

/// Arithmetic encoder. Feed bits with `encode_bit`, then call `finish`.
pub struct ArithEncoder {
    low: u64,        // logical low; bit 32 holds an outstanding carry
    range: u32,      // current range width
    cache: u8,       // last withheld byte (pending carry resolution)
    cache_size: u64, // number of withheld bytes (1 cache + run of 0xFF)
    out: Vec<u8>,
}

impl ArithEncoder {
    pub fn new() -> Self {
        // Canonical LZMA init: one pending (zero) cache byte. The decoder
        // reads — and discards — this leading byte to stay aligned.
        ArithEncoder {
            low: 0,
            range: 0xFFFF_FFFF,
            cache: 0,
            cache_size: 1,
            out: Vec::new(),
        }
    }

    /// Encode one bit. `prob_one` is the 12-bit probability that `bit == 1`,
    /// constrained to `1..=4095`. `bit` must be 0 or 1.
    pub fn encode_bit(&mut self, prob_one: u16, bit: u8) {
        // Always-on clamp (not just debug): an out-of-range probability from an
        // external model would otherwise spin the renormalize loop forever
        // (prob_one==0) or underflow `range` (prob_one>=4096) in release builds.
        let prob_one = prob_one.clamp(1, PROB_SCALE as u16 - 1);
        debug_assert!(bit <= 1);

        // `r1` is the sub-range assigned to bit==1; it sits at the bottom of
        // the interval. Both slices are non-zero because 1 <= prob_one <= 4095
        // and `range >= TOP` after every renormalization.
        let r1 = ((self.range as u64 * prob_one as u64) >> PROB_BITS) as u32;
        if bit == 1 {
            self.range = r1;
        } else {
            self.low += r1 as u64;
            self.range -= r1;
        }

        while self.range < TOP {
            self.shift_low();
            self.range <<= 8;
        }
    }

    fn shift_low(&mut self) {
        let low = self.low;
        // If the top byte is not 0xFF, or a carry has reached bit 32, we can
        // resolve all withheld bytes: cache + carry, then the run of 0xFF bytes
        // becomes 0x00 on carry (else stays 0xFF).
        if low < 0xFF00_0000u64 || low > 0xFFFF_FFFFu64 {
            let carry = (low >> 32) as u8; // 0 or 1
            let mut temp = self.cache;
            loop {
                self.out.push(temp.wrapping_add(carry));
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = ((low >> 24) & 0xFF) as u8;
        }
        // Always count this byte as pending (it becomes part of the next run).
        self.cache_size += 1;
        self.low = (low << 8) & 0xFFFF_FFFF;
    }

    /// Flush remaining state and return the encoded byte stream.
    pub fn finish(mut self) -> Vec<u8> {
        // Drain the full 32 bits of `low` (4 bytes) plus the cache.
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

impl Default for ArithEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Arithmetic decoder. Mirror of `ArithEncoder`; feed the same probabilities.
pub struct ArithDecoder<'a> {
    range: u32,
    code: u32,
    data: &'a [u8],
    pos: usize,
}

impl<'a> ArithDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = ArithDecoder {
            range: 0xFFFF_FFFF,
            code: 0,
            data,
            pos: 0,
        };
        // The encoder's first `shift_low` withholds one byte (cache) before any
        // real output, so the very first emitted byte is that leading cache
        // (0x00). Prime `code` with the next 4 stream bytes; the decoder reads
        // one extra leading byte to stay aligned with the encoder.
        d.next_byte(); // consume the leading cache byte
        for _ in 0..4 {
            d.code = (d.code << 8) | d.next_byte() as u32;
        }
        d
    }

    fn next_byte(&mut self) -> u8 {
        let b = if self.pos < self.data.len() {
            self.data[self.pos]
        } else {
            0
        };
        self.pos += 1;
        b
    }

    /// Decode one bit using the same `prob_one` the encoder used at this step.
    pub fn decode_bit(&mut self, prob_one: u16) -> u8 {
        // Must clamp identically to encode_bit to stay in lockstep.
        let prob_one = prob_one.clamp(1, PROB_SCALE as u16 - 1);

        let r1 = ((self.range as u64 * prob_one as u64) >> PROB_BITS) as u32;
        let bit;
        if self.code < r1 {
            bit = 1;
            self.range = r1;
        } else {
            bit = 0;
            self.code -= r1;
            self.range -= r1;
        }

        while self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | self.next_byte() as u32;
        }
        bit
    }
}

// --- Adaptive order-0 bit model: a self-contained correctness proof ---------
//
// Each of the 256 possible "tree contexts" (a node in the per-byte binary tree
// of already-seen high bits) carries one adaptive probability. The model is
// fed identically on both sides, so no table is transmitted.

const MAGIC: &[u8; 4] = b"ARB1";
const INIT_PROB: u16 = 2048; // p(1) = 1/2
const ADAPT_SHIFT: u32 = 5; // learning rate

struct BitModel {
    // 256 contexts: index 1 is the tree root, then 2*ctx+bit walks down. We
    // size for a full byte tree (indices 1..=255 used).
    probs: [u16; 256],
}

impl BitModel {
    fn new() -> Self {
        BitModel {
            probs: [INIT_PROB; 256],
        }
    }

    fn get(&self, ctx: usize) -> u16 {
        self.probs[ctx]
    }

    fn update(&mut self, ctx: usize, bit: u8) {
        let p = self.probs[ctx] as i32;
        let np = if bit == 1 {
            p + (((PROB_SCALE as i32) - p) >> ADAPT_SHIFT)
        } else {
            p - (p >> ADAPT_SHIFT)
        };
        // Clamp to the legal 1..=4095 range so the coder never sees p==0/4096.
        self.probs[ctx] = np.clamp(1, (PROB_SCALE as i32) - 1) as u16;
    }
}

/// Compress arbitrary bytes with an adaptive order-0 bit model. The output is a
/// 4-byte magic + 8-byte length header followed by the arithmetic stream.
/// Lossless.
pub fn compress_adaptive(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2 + 16);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    if data.is_empty() {
        return out;
    }

    let mut model = BitModel::new();
    let mut enc = ArithEncoder::new();
    for &byte in data {
        let mut ctx = 1usize; // tree root
        for i in (0..8).rev() {
            let bit = (byte >> i) & 1;
            let p = model.get(ctx);
            enc.encode_bit(p, bit);
            model.update(ctx, bit);
            ctx = (ctx << 1) | bit as usize;
            ctx &= 0xFF; // keep within the 256-entry table
        }
    }
    out.extend_from_slice(&enc.finish());
    out
}

/// Decompress a blob produced by `compress_adaptive`. The original length is
/// read from the header (`out_len` is ignored; pass 0 if unknown). Lossless.
pub fn decompress_adaptive(blob: &[u8], _out_len: usize) -> Result<Vec<u8>, String> {
    if blob.len() < 12 || &blob[0..4] != MAGIC {
        return Err("not an ARB1 stream".into());
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&blob[4..12]);
    let orig_len = u64::from_le_bytes(len_bytes) as usize;
    if orig_len == 0 {
        return Ok(Vec::new());
    }

    let mut model = BitModel::new();
    let mut dec = ArithDecoder::new(&blob[12..]);
    // Cap the up-front allocation so a crafted length header can't request an
    // aborting allocation; the loop still produces exactly orig_len bytes.
    let mut out = Vec::with_capacity(orig_len.min(1 << 20));
    for _ in 0..orig_len {
        let mut ctx = 1usize;
        let mut byte = 0u8;
        for _ in 0..8 {
            let p = model.get(ctx);
            let bit = dec.decode_bit(p);
            model.update(ctx, bit);
            byte = (byte << 1) | bit;
            ctx = (ctx << 1) | bit as usize;
            ctx &= 0xFF;
        }
        out.push(byte);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(data: &[u8]) {
        let blob = compress_adaptive(data);
        let back = decompress_adaptive(&blob, data.len()).expect("decompress");
        assert_eq!(back, data, "round-trip mismatch (len {})", data.len());
    }

    #[test]
    fn empty() {
        round_trip(&[]);
    }

    #[test]
    fn all_zero() {
        round_trip(&[0u8; 4096]);
    }

    #[test]
    fn single_byte_all_values() {
        for b in 0u16..=255 {
            round_trip(&[b as u8]);
        }
    }

    #[test]
    fn repeated_text() {
        let mut data = Vec::new();
        for _ in 0..500 {
            data.extend_from_slice(b"the quick brown fox jumps over the lazy dog. ");
        }
        round_trip(&data);
    }

    #[test]
    fn pseudo_random_lcg() {
        // Simple LCG (Numerical Recipes constants); no external crate.
        let mut state: u32 = 0x1234_5678;
        let mut data = Vec::with_capacity(50_000);
        for _ in 0..50_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            data.push((state >> 24) as u8);
        }
        round_trip(&data);
    }

    #[test]
    fn all_one_bits() {
        round_trip(&[0xFFu8; 4096]);
    }

    #[test]
    fn carry_stress() {
        // Long runs designed to provoke the 0xFF cache/carry path.
        let mut data = Vec::new();
        for i in 0..10_000u32 {
            data.push(if i % 7 == 0 { 0x00 } else { 0xFF });
        }
        round_trip(&data);
    }

    #[test]
    fn repetitive_is_much_smaller() {
        let mut data = Vec::new();
        for _ in 0..2000 {
            data.extend_from_slice(b"AAAAAAAAAAAAAAAA");
        }
        let blob = compress_adaptive(&data);
        // Highly repetitive input must compress dramatically (header is 12 B).
        assert!(
            blob.len() * 8 < data.len(),
            "expected strong compression: {} -> {}",
            data.len(),
            blob.len()
        );
        // And it must still decode exactly.
        let back = decompress_adaptive(&blob, data.len()).expect("decompress");
        assert_eq!(back, data);
    }
}
