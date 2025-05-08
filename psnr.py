# psnr.py
import numpy as np

def psnr(imagen_original, imagen_procesada):
    """
    Calcula el PSNR (Peak Signal to Noise Ratio) entre dos imágenes.
    """
    # Calcular el error cuadrático medio (MSE)
    mse = np.mean((imagen_original - imagen_procesada) ** 2)
    if mse == 0:  # Si no hay diferencia, PSNR es infinito
        return float('inf')
    # Calcular PSNR
    max_pixel = 255.0 
    psnr_value = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr_value
