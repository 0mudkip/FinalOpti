import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float
from gradiente import descenso_gradiente_simple, descenso_gradiente_momentum, descenso_gradiente_nesterov

#imagen = imread('prueba1.jpeg', as_gray=True)
#imagen = imread('prueba2.jpg', as_gray=True)
imagen = imread('prueba3.jpg', as_gray=True)
imagen = img_as_float(imagen)
imagen = imagen*255
nivel_ruido = 20
ruido = np.random.normal(0, nivel_ruido, imagen.shape)
imagen_ruidosa = imagen + ruido

plt.imshow(ruido, cmap='gray')
plt.show()


lambda_param = 0.1  
alpha = 0.1 
beta = 0.9  # Momentum
num_iter = 10000  # Número de iteraciones

u_inicial = np.copy(imagen_ruidosa)

# 1. Descenso de gradiente simple
u_simple = descenso_gradiente_simple(u_inicial, imagen_ruidosa, lambda_param, alpha, num_iter)

# 2. Descenso de gradiente con momentum
u_momentum = descenso_gradiente_momentum(u_inicial, imagen_ruidosa, lambda_param, alpha, beta, num_iter)

# 3. Descenso de gradiente con Nesterov
u_nesterov = descenso_gradiente_nesterov(u_inicial, imagen_ruidosa, lambda_param, alpha, beta, num_iter)

# Mostrar resultados
plt.figure(figsize=(15,10))

# Imagen original
plt.subplot(2, 3, 1)
plt.title('Imagen Original')
plt.imshow(imagen, cmap='gray')
plt.axis('off')

# Imagen con ruido
plt.subplot(2, 3, 2)
plt.title('Imagen Ruidosa')
plt.imshow(imagen_ruidosa, cmap='gray')
plt.axis('off')

# Resultado Descenso Gradiente Simple
plt.subplot(2, 3, 3)
plt.title('Denoising: Gradiente Simple')
plt.imshow(u_simple, cmap='gray')
plt.axis('off')

# Resultado Descenso Gradiente con Momentum
plt.subplot(2, 3, 4)
plt.title('Denoising: Momentum')
plt.imshow(u_momentum, cmap='gray')
plt.axis('off')

# Resultado Descenso Gradiente con Nesterov
plt.subplot(2, 3, 5)
plt.title('Denoising: Nesterov')
plt.imshow(u_nesterov, cmap='gray')
plt.axis('off')

plt.show()
