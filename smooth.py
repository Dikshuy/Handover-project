import numpy as np
import matplotlib.pyplot as plt


def blending_weights(t, t0, alpha=0.5):
    lambda_h = 1 / (1 + np.exp(-alpha * (t - t0/2)))
    lambda_a = 1 - lambda_h
    return lambda_h, lambda_a

def change_weights(t, T):
    lambda_h = t/T
    lambda_a = 1 - lambda_h
    return lambda_h, lambda_a

T = 20  # takeover time allowed
a = np.linspace(0, T, 100)
lambda_h, lambda_a = blending_weights(a, T)
lambda_hc, lambda_ac = change_weights(a, T)

plt.plot(a, lambda_a, label = "AV")
plt.plot(a, lambda_h, label = "Human")
plt.legend()
plt.show()

plt.plot(a, lambda_ac, label = "AV")
plt.plot(a, lambda_hc, label = "Human")
plt.legend()
plt.show()