import numpy as np
import matplotlib.pyplot as plt


def blending_weights(t, t0=10, alpha=0.5):
    lambda_h = 1 / (1 + np.exp(-alpha * (t - t0)))
    lambda_a = 1 - lambda_h
    return lambda_h, lambda_a

def change_weights(t):
    lambda_h = t/20
    lambda_a = 1 - lambda_h
    return lambda_h, lambda_a

a = np.linspace(0, 20, 100)
lambda_h, lambda_a = blending_weights(a)
lambda_hc, lambda_ac = change_weights(a)

plt.plot(a, lambda_a, label = "AV")
plt.plot(a, lambda_h, label = "Human")
plt.legend()
plt.show()

plt.plot(a, lambda_ac, label = "AV")
plt.plot(a, lambda_hc, label = "Human")
plt.legend()
plt.show()