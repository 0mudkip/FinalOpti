import autograd.numpy as np
from autograd import grad

# Definimos la función objetivo
def objetivo(u, f, lambda_param):
    grad_u = np.gradient(u) 
    grad_u = np.array(grad_u)  
    term_1 = 0.5 * np.sum((u - f)**2)  # 1/2 ||u - f||^2
    term_2 = 0.5 * lambda_param * np.sum(grad_u**2)  # 1/2 λ ||∇u||^2
    return term_1 + term_2

grad_objetivo = grad(objetivo)

def descenso_gradiente_simple(u, f, lambda_param, alpha, num_iter, tol=1e-9):
    prev_u = u.copy()  # Copiar la imagen inicial
    for i in range(num_iter):
        gradiente = grad_objetivo(u, f, lambda_param)
        u = u - alpha * gradiente 
        diff = np.linalg.norm(u - prev_u) 
        # Si la diferencia es menor que el umbral de tolerancia, detener el algoritmo
        if diff < tol:
            print(f"Convergencia alcanzada en la iteración {i+1}")
            break

        prev_u = u.copy()  # Actualizar la imagen anterior para la siguiente iteración
    
    return u



def descenso_gradiente_momentum(u, f, lambda_param, alpha, beta, num_iter, tol=1e-9):
    v = np.zeros_like(u)  # Inicializar el vector de velocidad
    prev_u = u.copy()  # Copiar la imagen inicial
    for i in range(num_iter):
        gradiente = grad_objetivo(u, f, lambda_param)
        v = beta * v + (1 - beta) * gradiente  # Actualización del momentum
        u = u - alpha * v  
        diff = np.linalg.norm(u - prev_u) 
        # Si la diferencia es menor que el umbral de tolerancia, detener el algoritmo
        if diff < tol:
            print(f"Convergencia alcanzada en la iteración {i+1}")
            break

        prev_u = u.copy()  # Actualizar la imagen anterior para la siguiente iteración
    
    return u


def descenso_gradiente_nesterov(u, f, lambda_param, alpha, beta, num_iter, tol=1e-9):
    v = np.zeros_like(u)  # Inicializar el vector de velocidad
    prev_u = u.copy()  # Copiar la imagen inicial
    for i in range(num_iter):
        u_temp = u - beta * v  # Imagen "temporal" con aceleración de Nesterov
        gradiente = grad_objetivo(u_temp, f, lambda_param)
        v = beta * v + (1 - beta) * gradiente  # Actualización del momentum
        u = u - alpha * v  # Actualización de la imagen
        diff = np.linalg.norm(u - prev_u)  # Norm de la diferencia entre iteraciones

        # Si la diferencia es menor que el umbral de tolerancia, detener el algoritmo
        if diff < tol:
            print(f"Convergencia alcanzada en la iteración {i+1}")
            break

        prev_u = u.copy()  # Actualizar la imagen anterior para la siguiente iteración
    
    return u
