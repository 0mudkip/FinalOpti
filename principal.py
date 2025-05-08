# main.py
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float
from psnr import psnr  # Importar la función psnr desde el archivo psnr.py
from gradiente import descenso_gradiente_simple, descenso_gradiente_momentum, descenso_gradiente_nesterov

# Cargar la imagen
imagen = imread('prueba3.jpg', as_gray=True)
imagen = img_as_float(imagen)
imagen = imagen * 255  # Escalar a 255

# Nivel de ruido 10
nivel_ruido_10 = 10
ruido_10 = np.random.normal(0, nivel_ruido_10, imagen.shape)
imagen_ruidosa_10 = imagen + ruido_10

# Nivel de ruido 20
nivel_ruido_20 = 20
ruido_20 = np.random.normal(0, nivel_ruido_20, imagen.shape)
imagen_ruidosa_20 = imagen + ruido_20

# Nivel de ruido 50
nivel_ruido_50 = 50  # Nivel de ruido cambiado a 50
ruido_50 = np.random.normal(0, nivel_ruido_50, imagen.shape)
imagen_ruidosa_50 = imagen + ruido_50

# Parámetros
lambda_param = 0.5  
alpha = 0.1 
beta = 0.9  # Momentum
num_iter = 10000  # Número de iteraciones

# Mostrar resultados para nivel de ruido 10
u_inicial_10 = np.copy(imagen_ruidosa_10)

u_simple_10 = descenso_gradiente_simple(u_inicial_10, imagen_ruidosa_10, lambda_param, alpha, num_iter)
u_momentum_10 = descenso_gradiente_momentum(u_inicial_10, imagen_ruidosa_10, lambda_param, alpha, beta, num_iter)
u_nesterov_10 = descenso_gradiente_nesterov(u_inicial_10, imagen_ruidosa_10, lambda_param, alpha, beta, num_iter)

# Calcular PSNR para el nivel de ruido 10
psnr_simple_10 = psnr(imagen, u_simple_10)
psnr_momentum_10 = psnr(imagen, u_momentum_10)
psnr_nesterov_10 = psnr(imagen, u_nesterov_10)

print(f"PSNR (Gradiente Simple) - Nivel 10: {psnr_simple_10} dB")
print(f"PSNR (Momentum) - Nivel 10: {psnr_momentum_10} dB")
print(f"PSNR (Nesterov) - Nivel 10: {psnr_nesterov_10} dB")

# Mostrar resultados para nivel de ruido 20
u_inicial_20 = np.copy(imagen_ruidosa_20)

u_simple_20 = descenso_gradiente_simple(u_inicial_20, imagen_ruidosa_20, lambda_param, alpha, num_iter)
u_momentum_20 = descenso_gradiente_momentum(u_inicial_20, imagen_ruidosa_20, lambda_param, alpha, beta, num_iter)
u_nesterov_20 = descenso_gradiente_nesterov(u_inicial_20, imagen_ruidosa_20, lambda_param, alpha, beta, num_iter)

# Calcular PSNR para el nivel de ruido 20
psnr_simple_20 = psnr(imagen, u_simple_20)
psnr_momentum_20 = psnr(imagen, u_momentum_20)
psnr_nesterov_20 = psnr(imagen, u_nesterov_20)

print(f"PSNR (Gradiente Simple) - Nivel 20: {psnr_simple_20} dB")
print(f"PSNR (Momentum) - Nivel 20: {psnr_momentum_20} dB")
print(f"PSNR (Nesterov) - Nivel 20: {psnr_nesterov_20} dB")

# Mostrar resultados para nivel de ruido 50
u_inicial_50 = np.copy(imagen_ruidosa_50)

u_simple_50 = descenso_gradiente_simple(u_inicial_50, imagen_ruidosa_50, lambda_param, alpha, num_iter)
u_momentum_50 = descenso_gradiente_momentum(u_inicial_50, imagen_ruidosa_50, lambda_param, alpha, beta, num_iter)
u_nesterov_50 = descenso_gradiente_nesterov(u_inicial_50, imagen_ruidosa_50, lambda_param, alpha, beta, num_iter)

# Calcular PSNR para el nivel de ruido 50
psnr_simple_50 = psnr(imagen, u_simple_50)
psnr_momentum_50 = psnr(imagen, u_momentum_50)
psnr_nesterov_50 = psnr(imagen, u_nesterov_50)

print(f"PSNR (Gradiente Simple) - Nivel 50: {psnr_simple_50} dB")
print(f"PSNR (Momentum) - Nivel 50: {psnr_momentum_50} dB")
print(f"PSNR (Nesterov) - Nivel 50: {psnr_nesterov_50} dB")

# Mostrar gráficas para nivel de ruido 10
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.title('Imagen Original - Nivel 10')
plt.imshow(imagen, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Imagen Ruidosa - Nivel 10')
plt.imshow(imagen_ruidosa_10, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Denoising: Gradiente Simple - Nivel 10')
plt.imshow(u_simple_10, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Denoising: Momentum - Nivel 10')
plt.imshow(u_momentum_10, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Denoising: Nesterov - Nivel 10')
plt.imshow(u_nesterov_10, cmap='gray')
plt.axis('off')

plt.show()

# Mostrar gráficas para nivel de ruido 20
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.title('Imagen Original - Nivel 20')
plt.imshow(imagen, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Imagen Ruidosa - Nivel 20')
plt.imshow(imagen_ruidosa_20, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Denoising: Gradiente Simple - Nivel 20')
plt.imshow(u_simple_20, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Denoising: Momentum - Nivel 20')
plt.imshow(u_momentum_20, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Denoising: Nesterov - Nivel 20')
plt.imshow(u_nesterov_20, cmap='gray')
plt.axis('off')

plt.show()

# Mostrar gráficas para nivel de ruido 50
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.title('Imagen Original - Nivel 50')
plt.imshow(imagen, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Imagen Ruidosa - Nivel 50')
plt.imshow(imagen_ruidosa_50, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Denoising: Gradiente Simple - Nivel 50')
plt.imshow(u_simple_50, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Denoising: Momentum - Nivel 50')
plt.imshow(u_momentum_50, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Denoising: Nesterov - Nivel 50')
plt.imshow(u_nesterov_50, cmap='gray')
plt.axis('off')

plt.show()
