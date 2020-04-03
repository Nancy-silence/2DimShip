import numpy as np
import matplotlib.pyplot as plt

plt.ion()
for i in range(200):
    plt.cla()
    x = np.linspace(0, i + 1, 1000)
    y = np.sin(x)
    plt.plot(x, y)
    plt.pause(0.1)
plt.show()
plt.ioff()