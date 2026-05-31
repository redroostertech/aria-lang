//! End-to-end predictive ("neural") lossless compressor.
//!
//! This is the architecture DeepMind pointed at with "Language Modeling Is
//! Compression": a predictor estimates P(next bit), an arithmetic coder turns
//! that estimate into a fraction of a bit, and the decoder — fed the *identical*
//! predictions, because the model is driven only by already-decoded data —
//! recovers the original exactly. Lossless: data quality is fully preserved.
//!
//! Here the predictor is the context-mixing model in `predict`. Swapping it for
//! a transformer forward pass (from `transformer`) is the future neural tier;
//! the codec wiring below does not change.

use crate::arith::{ArithDecoder, ArithEncoder};
use crate::predict::Predictor;

const MAGIC: &[u8; 4] = b"ARN1";

/// Compress arbitrary bytes. Output = header + arithmetic stream. Lossless.
pub fn compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2 + 16);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    if data.is_empty() {
        return out;
    }

    let mut model = Predictor::new();
    let mut enc = ArithEncoder::new();
    for &byte in data {
        // MSB-first; the predictor and coder stay in lockstep because each
        // `predict_bit` is followed by `update_bit` with the actual bit.
        for k in (0..8).rev() {
            let bit = (byte >> k) & 1;
            let p = model.predict_bit();
            enc.encode_bit(p, bit);
            model.update_bit(bit);
        }
    }
    out.extend_from_slice(&enc.finish());
    out
}

/// Decompress a blob produced by `compress`. Lossless inverse.
pub fn decompress(blob: &[u8]) -> Result<Vec<u8>, String> {
    if blob.len() < 12 || &blob[0..4] != MAGIC {
        return Err("not an ARN1 stream".into());
    }
    let mut lb = [0u8; 8];
    lb.copy_from_slice(&blob[4..12]);
    let len = u64::from_le_bytes(lb) as usize;
    if len == 0 {
        return Ok(Vec::new());
    }

    let mut model = Predictor::new();
    let mut dec = ArithDecoder::new(&blob[12..]);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let mut byte = 0u8;
        for _ in 0..8 {
            let p = model.predict_bit();
            let bit = dec.decode_bit(p);
            model.update_bit(bit);
            byte = (byte << 1) | bit;
        }
        out.push(byte);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(data: &[u8]) {
        let packed = compress(data);
        let back = decompress(&packed).expect("decompress");
        assert_eq!(back, data, "round-trip mismatch (len {})", data.len());
    }

    #[test]
    fn empty() {
        roundtrip(b"");
    }

    #[test]
    fn short_text() {
        roundtrip(b"the quick brown fox jumps over the lazy dog");
    }

    #[test]
    fn repetitive_text_compresses() {
        let mut data = Vec::new();
        for _ in 0..300 {
            data.extend_from_slice(b"the quick brown fox jumps over the lazy dog. ");
        }
        let packed = compress(&data);
        roundtrip(&data);
        // Predictive model should crush repetitive text far past order-0.
        assert!(packed.len() * 4 < data.len(), "expected strong compression: {} -> {}", data.len(), packed.len());
    }

    #[test]
    fn pseudo_random_roundtrips() {
        // Simple LCG; no rand crate.
        let mut state: u32 = 0xCAFE_1234;
        let mut data = Vec::with_capacity(4000);
        for _ in 0..4000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            data.push((state >> 24) as u8);
        }
        roundtrip(&data);
    }
}
