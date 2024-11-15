import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz, dlti, dimpulse


# Custom zplane function
def zplane(b, a):
    from numpy import roots

    zeros = roots(b)  # Calculate zeros
    poles = roots(a)  # Calculate poles

    plt.figure()
    plt.scatter(zeros.real, zeros.imag, s=50, label='Zeros', marker='o', facecolors='none', edgecolors='blue')
    plt.scatter(poles.real, poles.imag, s=50, label='Poles', marker='x', color='red')

    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k--')  # Unit circle

    plt.axvline(0, color='black', lw=1)
    plt.axhline(0, color='black', lw=1)
    plt.title('Poles and Zeros')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

    return zeros, poles


# Coefficients
A = [0.090909]
B = [1, -0.90909]

# Sampling
T = 100e-6
Fe = 1 / T
Te = T
t = np.arange(0, 200 * Te, Te)
x = np.ones(len(t))

# Impulse Response
system = dlti(A, B)
t_imp, imp_response = dimpulse(system, n=200)
imp_response = np.squeeze(imp_response)

plt.figure(1)
plt.stem(t_imp, imp_response, use_line_collection=True)
plt.title('Impulse Response')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.grid()

# Frequency Response
w, h = freqz(A, B, worN=512)

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(w / np.pi * Fe / 2, 20 * np.log10(abs(h)))
plt.title('Frequency Response (Gain and Phase)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(w / np.pi * Fe / 2, np.angle(h))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.grid()

# Filter Response
y = lfilter(A, B, x)

plt.figure(3)
plt.plot(t, y)
plt.title('Step Response')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()

# Poles and Zeros
zeros, poles = zplane(A, B)

plt.show()
