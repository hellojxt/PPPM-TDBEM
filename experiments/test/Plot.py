import matplotlib.pyplot as plt
import numpy as np

result = np.loadtxt("experiments/test/error.txt")
idx = np.zeros_like(result)

for i in range(0, len(idx)):
    idx[i] = i

plt.plot(idx, result)
plt.show()