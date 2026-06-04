def build(n):
    a = None
    while n: a = (n, a); n -= 1
    return a
def suml(xs):
    acc = 0
    while xs is not None: acc += xs[0]; xs = xs[1]
    return acc
print(suml(build(20000000)))
