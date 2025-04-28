import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.io import imread
from scipy.ndimage import sobel


def funcion_objetivo(u, f, lambda_reg):
    # Primer término: error de datos (diferencia entre u y f)
    data_term = 1/2 * np.sum((u - f)**2)

    # Gradiente de u
    grad_x = sobel(u, axis=0, mode='reflect')  # Derivada en x
    grad_y = sobel(u, axis=1, mode='reflect')  # Derivada en y
    grad_term = 0.5 * lambda_reg * np.sum(grad_x**2 + grad_y**2)

    # Función objetivo
    return data_term + grad_term

imagen = imread('prueba1.jpg', as_gray=True) 
imagen = img_as_float(imagen) 
nivel_ruido = 0.5  

# Crear ruido gaussiano
ruido = np.random.normal(0, nivel_ruido, imagen.shape)

# Sumar ruido a la imagen
imagen_ruidosa = imagen + ruido

# Clip para mantener los valores en [0, 1]
imagen_ruidosa = np.clip(imagen_ruidosa, 0, 1)

# Mostrar imagen original y ruidosa
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Imagen original')
plt.imshow(imagen, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Imagen con ruido')
plt.imshow(imagen_ruidosa, cmap='gray')
plt.axis('off')

plt.show()
