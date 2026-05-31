//! Structural ("type-aware") compression layer + benchmark.
//!
//! This is the *model* stage that sits in front of the rANS entropy coder.
//! Because Aria knows the shape of data, it can transform it into a form that
//! entropy-codes far better than raw bytes. Here we demonstrate the two
//! highest-impact transforms for tabular/columnar numeric data:
//!
//!   * columnar split  — group same-typed fields together so values cluster
//!   * delta + zig-zag — store differences between neighbours (tiny, repetitive)
//!
//! The result is then handed to `rans::compress`. This is exactly the strategy
//! Parquet/zstd-dictionary use to beat general byte compressors like zip, and
//! it is fully lossless — original data is reconstructed exactly.

use std::time::Instant;

use crate::rans;

fn zigzag(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

fn unzigzag(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

/// Delta-encode a column, zig-zag, and serialize as little-endian u64.
fn encode_column(col: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(col.len() * 8);
    let mut prev = 0i64;
    for &v in col {
        let delta = v.wrapping_sub(prev);
        out.extend_from_slice(&zigzag(delta).to_le_bytes());
        prev = v;
    }
    out
}

fn decode_column(bytes: &[u8], rows: usize) -> Vec<i64> {
    let mut col = Vec::with_capacity(rows);
    let mut prev = 0i64;
    for i in 0..rows {
        let off = i * 8;
        let mut b = [0u8; 8];
        b.copy_from_slice(&bytes[off..off + 8]);
        let delta = unzigzag(u64::from_le_bytes(b));
        let v = prev.wrapping_add(delta);
        col.push(v);
        prev = v;
    }
    col
}

/// Type-aware compression of a set of equal-length i64 columns.
/// Returns the compressed blob (row count + per-column rANS streams).
pub fn compress_columns(cols: &[Vec<i64>]) -> Vec<u8> {
    let rows = cols.first().map(|c| c.len()).unwrap_or(0);
    let mut out = Vec::new();
    out.extend_from_slice(&(cols.len() as u32).to_le_bytes());
    out.extend_from_slice(&(rows as u64).to_le_bytes());
    for col in cols {
        let transformed = encode_column(col);
        let packed = rans::compress(&transformed);
        out.extend_from_slice(&(packed.len() as u64).to_le_bytes());
        out.extend_from_slice(&packed);
    }
    out
}

pub fn decompress_columns(blob: &[u8]) -> Result<Vec<Vec<i64>>, String> {
    if blob.len() < 12 {
        return Err("truncated columnar blob".into());
    }
    let ncols = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]) as usize;
    let mut rb = [0u8; 8];
    rb.copy_from_slice(&blob[4..12]);
    let rows = u64::from_le_bytes(rb) as usize;

    let mut p = 12usize;
    let mut cols = Vec::with_capacity(ncols);
    for _ in 0..ncols {
        if p + 8 > blob.len() {
            return Err("truncated column length".into());
        }
        let mut lb = [0u8; 8];
        lb.copy_from_slice(&blob[p..p + 8]);
        let len = u64::from_le_bytes(lb) as usize;
        p += 8;
        if p + len > blob.len() {
            return Err("truncated column data".into());
        }
        let transformed = rans::decompress(&blob[p..p + len])?;
        p += len;
        // `rows` is attacker-controlled; guard the multiply and verify the
        // decompressed payload is large enough before decode_column indexes it.
        let need = rows
            .checked_mul(8)
            .ok_or("columnar row count overflows")?;
        if transformed.len() < need {
            return Err("column shorter than declared row count".into());
        }
        cols.push(decode_column(&transformed, rows));
    }
    Ok(cols)
}

// ---- benchmark ----------------------------------------------------------

/// Serialize columns row-major as raw i64 LE — the naive "array of structs"
/// layout a normal program would write to disk. This is what we compress
/// against, for a fair comparison.
fn raw_row_major(cols: &[Vec<i64>]) -> Vec<u8> {
    let rows = cols.first().map(|c| c.len()).unwrap_or(0);
    let mut out = Vec::with_capacity(rows * cols.len() * 8);
    for r in 0..rows {
        for col in cols {
            out.extend_from_slice(&col[r].to_le_bytes());
        }
    }
    out
}

/// Deterministic synthetic time-series: (timestamp, sensor_id, temperature).
/// Mimics real telemetry: timestamps tick up slowly, ids cycle, temps drift.
fn synthetic_dataset(rows: usize) -> Vec<Vec<i64>> {
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut next = || -> u64 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed >> 33
    };

    let mut ts = Vec::with_capacity(rows);
    let mut id = Vec::with_capacity(rows);
    let mut temp = Vec::with_capacity(rows);

    let mut t = 1_700_000_000i64;
    let mut temperature = 2150i64; // 21.50 C in centidegrees
    for r in 0..rows {
        t += 1 + (next() % 4) as i64; // small positive increments
        ts.push(t);
        id.push((r % 64) as i64); // 64 sensors, cycling
        let drift = (next() % 5) as i64 - 2; // -2..+2
        temperature += drift;
        temp.push(temperature);
    }
    vec![ts, id, temp]
}

fn gzip_size(data: &[u8]) -> Option<usize> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new("gzip")
        .arg("-9")
        .arg("-c")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .ok()?;
    // Write stdin on a separate thread so it can't deadlock against stdout:
    // for large inputs gzip blocks writing output while we block writing input.
    let mut stdin = child.stdin.take()?;
    let owned = data.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&owned);
        // stdin dropped here -> EOF for gzip
    });
    let out = child.wait_with_output().ok()?;
    let _ = writer.join();
    if out.status.success() {
        Some(out.stdout.len())
    } else {
        None
    }
}

fn pct(part: usize, whole: usize) -> f64 {
    100.0 * part as f64 / whole as f64
}

pub fn bench() {
    let rows = 200_000;
    println!("Aria compression benchmark");
    println!("  dataset: {} rows x 3 i64 columns (synthetic telemetry)\n", rows);

    let cols = synthetic_dataset(rows);
    let raw = raw_row_major(&cols);
    let raw_len = raw.len();

    // 1) gzip -9 on the raw bytes (the zip-class baseline).
    let t = Instant::now();
    let gz = gzip_size(&raw);
    let gz_ms = t.elapsed().as_secs_f64() * 1000.0;

    // 2) Aria rANS on the raw bytes (entropy only, byte-blind).
    let t = Instant::now();
    let rans_only = rans::compress(&raw);
    let rans_ms = t.elapsed().as_secs_f64() * 1000.0;

    // 3) Aria type-aware: columnar + delta + zig-zag, then rANS.
    let t = Instant::now();
    let structural = compress_columns(&cols);
    let struct_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Verify the type-aware path is lossless.
    let t = Instant::now();
    let restored = decompress_columns(&structural).expect("decompress");
    let decode_ms = t.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(restored, cols, "LOSSLESS CHECK FAILED");

    println!("  {:<28} {:>12}  {:>8}  {:>9}", "method", "bytes", "ratio", "time");
    println!("  {:-<28} {:->12}  {:->8}  {:->9}", "", "", "", "");
    println!(
        "  {:<28} {:>12}  {:>7.1}%  {:>9}",
        "raw (i64 row-major)", raw_len, 100.0, "-"
    );
    match gz {
        Some(g) => println!(
            "  {:<28} {:>12}  {:>7.1}%  {:>7.1}ms",
            "gzip -9 (zip-class)", g, pct(g, raw_len), gz_ms
        ),
        None => println!("  {:<28} {:>12}", "gzip -9 (unavailable)", "-"),
    }
    println!(
        "  {:<28} {:>12}  {:>7.1}%  {:>7.1}ms",
        "Aria rANS (entropy only)",
        rans_only.len(),
        pct(rans_only.len(), raw_len),
        rans_ms
    );
    println!(
        "  {:<28} {:>12}  {:>7.1}%  {:>7.1}ms",
        "Aria type-aware + rANS",
        structural.len(),
        pct(structural.len(), raw_len),
        struct_ms
    );
    println!(
        "\n  type-aware decode (lossless verified): {:.1}ms",
        decode_ms
    );

    if let Some(g) = gz {
        let win = g as f64 / structural.len() as f64;
        println!(
            "  => Aria type-aware is {:.2}x smaller than gzip -9 on this data.",
            win
        );
    }
}
