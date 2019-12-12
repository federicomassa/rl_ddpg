import matplotlib.pyplot as plt
import numpy as np

tau = 0.05
mu = 0.0
sigma = 1.

dt = 0.001
T = 1.
n = int(T/dt)
t = np.linspace(0., T, n)

sigma_bis = sigma*np.sqrt(2./tau)
sqrtdt = np.sqrt(dt)

trials = 1000

X = np.zeros((trials, n))

# Ornstein-Uhlenbeck
for t in range(trials):
    x = X[t]
    for i in range(n-1):
        x[i+1] = x[i] + dt*(mu - x[i])/tau + sigma_bis*sqrtdt*np.random.randn()

# White gaussian noise
white_noise = np.zeros(n)
for i in range(n-1):
    white_noise[i+1] = sigma_bis*sqrtdt*np.random.randn()

averages = np.zeros(trials)
variances = np.zeros(trials)

for i in range(trials):
    averages[i] = X[i].mean()
    variances[i] = X[i].var()*n/(n-1)

plt.figure(0)
plt.hist(averages)

plt.figure(1)
plt.hist(variances)

plt.figure(2)
plt.plot(X[0])

plt.figure(3)
plt.plot(white_noise)

print("Mean: {}, Var: {}".format(averages.mean(), variances.mean()))

plt.show()