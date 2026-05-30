//! rANS (range Asymmetric Numeral System) entropy coder.
//!
//! This is the "back end" of Aria's compression engine — the part that turns a
//! probability model into bits at (very close to) the theoretical entropy
//! limit, while decoding faster than Huffman. It is the same coder that powers
//! Zstandard/FSE.
//!
//! This module implements a *static order-0* model: it measures the frequency
//! of each byte, normalizes those frequencies to a power-of-two total, and
//! codes against them. Smarter models (context modeling, neural prediction)
//! can be layered on top later — they only change how `freqs` is produced; the
//! coder below is unchanged.

const SCALE_BITS: u32 = 12;
const M: u32 = 1 << SCALE_BITS; // total frequency, must be a power of two
const RANS_BYTE_L: u32 = 1 << 23; // lower bound of the normalized interval

const MAGIC: &[u8; 4] = b"ARZ1";

/// Compress arbitrary bytes. Output = header + rANS stream. Lossless.
pub fn compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2 + 32);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());

    if data.is_empty() {
        return out;
    }

    let counts = count_freqs(data);
    let freqs = normalize(&counts);
    // Store the normalized table (256 * u16) so the decoder can rebuild it.
    for f in freqs.iter() {
        out.extend_from_slice(&(*f as u16).to_le_bytes());
    }

    let cum = build_cum(&freqs);
    let stream = encode(data, &freqs, &cum);
    out.extend_from_slice(&stream);
    out
}

/// Decompress a blob produced by `compress`. Lossless inverse.
pub fn decompress(blob: &[u8]) -> Result<Vec<u8>, String> {
    if blob.len() < 12 || &blob[0..4] != MAGIC {
        return Err("not an ARZ1 stream".into());
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&blob[4..12]);
    let orig_len = u64::from_le_bytes(len_bytes) as usize;

    if orig_len == 0 {
        return Ok(Vec::new());
    }

    let table_start = 12;
    let table_end = table_start + 256 * 2;
    if blob.len() < table_end {
        return Err("truncated frequency table".into());
    }
    let mut freqs = [0u32; 256];
    for i in 0..256 {
        let off = table_start + i * 2;
        freqs[i] = u16::from_le_bytes([blob[off], blob[off + 1]]) as u32;
    }

    let cum = build_cum(&freqs);
    let slot2sym = build_slot2sym(&freqs, &cum);
    let stream = &blob[table_end..];
    decode(stream, &freqs, &cum, &slot2sym, orig_len)
}

fn count_freqs(data: &[u8]) -> [u32; 256] {
    let mut c = [0u32; 256];
    for &b in data {
        c[b as usize] += 1;
    }
    c
}

/// Scale raw counts so present symbols have freq >= 1 and the total is exactly M.
fn normalize(counts: &[u32; 256]) -> [u32; 256] {
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    let mut freqs = [0u32; 256];
    if total == 0 {
        return freqs;
    }
    let mut sum: u32 = 0;
    for i in 0..256 {
        if counts[i] == 0 {
            continue;
        }
        let mut f = ((counts[i] as u64 * M as u64) / total) as u32;
        if f == 0 {
            f = 1;
        }
        freqs[i] = f;
        sum += f;
    }
    // Reconcile to exactly M.
    if sum > M {
        let mut excess = sum - M;
        while excess > 0 {
            // Take from the currently-largest symbol, keeping it >= 1.
            let mut mi = 0usize;
            let mut mv = 0u32;
            for i in 0..256 {
                if freqs[i] > mv {
                    mv = freqs[i];
                    mi = i;
                }
            }
            let take = excess.min(freqs[mi].saturating_sub(1));
            if take == 0 {
                break;
            }
            freqs[mi] -= take;
            excess -= take;
        }
    } else if sum < M {
        // Give the slack to the largest symbol.
        let mut mi = 0usize;
        let mut mv = 0u32;
        for i in 0..256 {
            if freqs[i] > mv {
                mv = freqs[i];
                mi = i;
            }
        }
        freqs[mi] += M - sum;
    }
    freqs
}

fn build_cum(freqs: &[u32; 256]) -> [u32; 257] {
    let mut cum = [0u32; 257];
    for i in 0..256 {
        cum[i + 1] = cum[i] + freqs[i];
    }
    cum
}

fn build_slot2sym(freqs: &[u32; 256], cum: &[u32; 257]) -> Vec<u8> {
    let mut table = vec![0u8; M as usize];
    for s in 0..256 {
        let start = cum[s] as usize;
        let end = (cum[s] + freqs[s]) as usize;
        for slot in start..end {
            table[slot] = s as u8;
        }
    }
    table
}

fn encode(data: &[u8], freqs: &[u32; 256], cum: &[u32; 257]) -> Vec<u8> {
    let cap = data.len() * 2 + 64;
    let mut buf = vec![0u8; cap];
    let mut pos = cap;
    let mut x: u32 = RANS_BYTE_L;

    // Symbols are encoded in reverse; rANS is last-in-first-out.
    for &b in data.iter().rev() {
        let f = freqs[b as usize];
        let c = cum[b as usize];
        let x_max = ((RANS_BYTE_L >> SCALE_BITS) << 8) * f;
        while x >= x_max {
            pos -= 1;
            buf[pos] = (x & 0xff) as u8;
            x >>= 8;
        }
        x = ((x / f) << SCALE_BITS) + (x % f) + c;
    }

    // Flush the 32-bit state so it reads back little-endian at the front.
    pos -= 1;
    buf[pos] = (x >> 24) as u8;
    pos -= 1;
    buf[pos] = (x >> 16) as u8;
    pos -= 1;
    buf[pos] = (x >> 8) as u8;
    pos -= 1;
    buf[pos] = x as u8;

    buf[pos..].to_vec()
}

fn decode(
    stream: &[u8],
    freqs: &[u32; 256],
    cum: &[u32; 257],
    slot2sym: &[u8],
    out_len: usize,
) -> Result<Vec<u8>, String> {
    if stream.len() < 4 {
        return Err("truncated rANS stream".into());
    }
    let mut x = (stream[0] as u32)
        | ((stream[1] as u32) << 8)
        | ((stream[2] as u32) << 16)
        | ((stream[3] as u32) << 24);
    let mut p = 4usize;
    let mut out = Vec::with_capacity(out_len);

    for _ in 0..out_len {
        let slot = x & (M - 1);
        let s = slot2sym[slot as usize];
        let f = freqs[s as usize];
        let c = cum[s as usize];
        x = f * (x >> SCALE_BITS) + slot - c;
        while x < RANS_BYTE_L {
            if p >= stream.len() {
                return Err("unexpected end of rANS stream".into());
            }
            x = (x << 8) | (stream[p] as u32);
            p += 1;
        }
        out.push(s);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(data: &[u8]) {
        let packed = compress(data);
        let back = decompress(&packed).expect("decompress");
        assert_eq!(back, data);
    }

    #[test]
    fn empty() {
        roundtrip(b"");
    }

    #[test]
    fn single_symbol() {
        roundtrip(&[7u8; 1000]);
    }

    #[test]
    fn text() {
        roundtrip(b"the quick brown fox jumps over the lazy dog, repeatedly and often");
    }

    #[test]
    fn all_bytes() {
        let data: Vec<u8> = (0..=255u8).cycle().take(10000).collect();
        roundtrip(&data);
    }
}
