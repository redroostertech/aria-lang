def steps(n):
    c = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        c += 1
    return c
total = 0
for i in range(1, 1000001): total += steps(i)
print(total)
