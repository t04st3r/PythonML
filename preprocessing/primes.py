import matplotlib.pyplot as plt
import math

l = []
r = range(4, 10000 + 1)
for num in r:
    for i in range(2, int(math.sqrt(num)) + 1):
        if (num % i) == 0:
            break
    else:
        l.append(num)

print(l)

a = []
b = []
for p in l:
    n = int(((p * p) - 1) / 24)
    a.append(n)
    n_str = str(int(n))
    b.append(int(n_str[-1]))

print(a)
print(b)

plt.scatter(l, a)
plt.axis([0, max(l), 0, max(a)])
plt.show()
