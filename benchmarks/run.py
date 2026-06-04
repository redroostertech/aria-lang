# Reproduces the README "Performance" table.
#   1. builds each Aria benchmark to a native binary (aria native -> cc -O2)
#   2. verifies aria/python/node produce identical output
#   3. reports best-of-RUNS wall-clock per language
# Run from the repo root:  python3 benchmarks/run.py
import subprocess, time, sys, os
BENCH = ["fib", "loopsum", "collatz", "listsum"]
RUNS = 3
ARIA = "./target/release/aria"

def build_and_verify():
    if not os.path.exists(ARIA):
        sys.exit("build the compiler first: cargo build --release")
    for b in BENCH:
        subprocess.run([ARIA, "native", f"benchmarks/{b}.aria", f"benchmarks/{b}.bin"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        # Aria's native main() prints the computed value, then echoes its own
        # Int return value as a trailing line; compare the computed first line.
        outs = {l: subprocess.run(cmd(l, b), capture_output=True, text=True).stdout.strip().splitlines()[0]
                for l in ("aria", "python", "node")}
        if len(set(outs.values())) != 1:
            sys.exit(f"output mismatch on {b}: {outs}")
    print("outputs verified identical across aria/python/node\n")

def cmd(lang, b):
    return {"aria":[f"./benchmarks/{b}.bin"],
            "python":["python3", f"benchmarks/{b}.py"],
            "node":["node", f"benchmarks/{b}.js"]}[lang]
def best(c):
    ts=[]
    for _ in range(RUNS):
        t0=time.perf_counter()
        subprocess.run(c, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        ts.append(time.perf_counter()-t0)
    return min(ts)*1000
build_and_verify()
r={b:{l:best(cmd(l,b)) for l in ("aria","python","node")} for b in BENCH}
print(f"{'benchmark':<10}{'aria(ms)':>10}{'python(ms)':>12}{'node(ms)':>10}{'py/aria':>9}{'node/aria':>10}")
for b in BENCH:
    a,p,n=r[b]["aria"],r[b]["python"],r[b]["node"]
    print(f"{b:<10}{a:>10.1f}{p:>12.1f}{n:>10.1f}{p/a:>8.1f}x{n/a:>9.1f}x")
print("DONE")
