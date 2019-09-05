import numpy as np
import matplotlib.pyplot as plt

def step(ts):
    return 1 * (ts > 0)

ts = np.linspace(-0.1, 0.1, 20) # Simulation time
Fuerza=step(ts)


plt.plot(ts,Fuerza)
plt.show()