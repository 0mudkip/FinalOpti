# psnr.py
import numpy as np

def psnr(imagen_original, imagen_procesada):
    """
    Calcula el PSNR (Peak Signal to Noise Ratio) entre dos imágenes.
    
    Args:
        imagen_original (numpy.ndarray): La imagen original.
        imagen_procesada (numpy.ndarray): La imagen procesada o denoised.
    
    Returns:
        float: El valor de PSNR en decibelios (dB).
    """
    # Calcular el error cuadrático medio (MSE)
    mse = np.mean((imagen_original - imagen_procesada) ** 2)
    if mse == 0:  # Si no hay diferencia, PSNR es infinito
        return float('inf')
    # Calcular PSNR
    max_pixel = 255.0  # Asumiendo imágenes de 8 bits
    psnr_value = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr_value
