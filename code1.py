import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz, impz, zplane

# Définition de la période T et de la fréquence d'échantillonnage Fe
T = 100e-6  # Période d'échantillonnage en secondes (100 microsecondes)
Fe = 1 / T  # Fréquence d'échantillonnage en Hz

# Définition des coefficients du filtre RII
A = [0.090909]               # Coefficient du numérateur (filtre RII)
B = [1, -0.90909]           # Coefficients du dénominateur (filtre RII)

# Définition de l'intervalle de temps
Te = T                        # Période d'échantillonnage
t = np.arange(0, 200 * Te, Te)  # Intervalle du temps (de 0 à 200ms)
x = np.ones(len(t))          # Entrée du filtre (Échelon Unitaire)

# Affichage de la réponse impulsionnelle du filtre
plt.figure(1)                # Créer une nouvelle figure pour la réponse impulsionnelle
impulse_response = impz(A, B)
plt.plot(impulse_response[0], impulse_response[1])  # Tracer la réponse impulsionnelle
plt.title('La Réponse Impulsionnelle')
plt.xlabel('Échantillons')
plt.ylabel('Amplitude')
plt.grid()

# Affichage de la réponse fréquentielle du filtre
plt.figure(2)                # Créer une nouvelle figure pour la réponse fréquentielle
w, h = freqz(A, B, worN=512) # Calcul de la réponse fréquentielle
plt.subplot(2, 1, 1)
plt.plot(w / np.pi * Fe / 2, 20 * np.log10(abs(h)))  # Tracer le gain en dB
plt.title('La Réponse Fréquentielle (Gain en dB & Phase)')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Gain (dB)')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(w / np.pi * Fe / 2, np.angle(h))  # Tracer la phase
plt_ylabel('Phase (radians)')
plt.xlabel('Fréquence (Hz)')
plt.grid()

# Calcul et affichage de la réponse fréquentielle
plt.figure(3)                # Créer une nouvelle figure pour la magnitude de la réponse
plt.plot(w / np.pi * Fe / 2, abs(h))  # Tracer la magnitude de la réponse fréquentielle
plt.title('Le Gabarit')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')
plt.grid()

# Filtrage du signal x à partir de la fonction de transfert
y = lfilter(A, B, x)       # Filtrer le signal x à l'aide du filtre

# Affichage de la réponse indicielle
plt.figure(4)                # Créer une nouvelle figure
plt.plot(t, y)              # Tracer la réponse indicielle
plt.title('La Réponse Indicielle')
plt.xlabel('Temps (s)')     # Étiquetage de l'axe des x
plt.ylabel('Amplitude')      # Étiquetage de l'axe des y
plt.grid()

# Visualisation des pôles et zéros du filtre
plt.figure(5)                # Créer une nouvelle figure pour la représentation des pôles et zéros
plt.subplot(1, 1, 1)  
z, p, k = zplane(A, B)      # Tracer le diagramme des pôles et zéros (l'utilisation de zplane nécessite une fonction personnalisée)
plt.title('Le Cercle de Plan Z [Im, Re]')
plt.grid()

plt.show()                  # Afficher toutes les figures