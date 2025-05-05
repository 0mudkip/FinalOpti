import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.io import imread
from scipy.ndimage import sobel


imagen = imread('prueba2.jpg', as_gray=True) 
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
