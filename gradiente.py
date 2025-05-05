import autograd.numpy as np
from autograd import grad

def objetivo(u, f, lambda_param):
    grad_u = np.gradient(u)  
    grad_u = np.array(grad_u)
    term_1 = 0.5 * np.sum((u - f)**2)  # 1/2 ||u - f||^2
    term_2 = 0.5 * lambda_param * np.sum(grad_u**2)  # 1/2 λ ||∇u||^2
    return term_1 + term_2


grad_objetivo = grad(objetivo)

def descenso_gradiente_simple(u, f, lambda_param, alpha, num_iter, tol=1e-6):
    prev_u = u.copy()  # Copiar la imagen inicial
    for i in range(num_iter):
        gradiente = grad_objetivo(u, f, lambda_param)
        u = u - alpha * gradiente 
        diff = np.linalg.norm(u - prev_u) 
        print(f"Iteración {i+1}, Diferencia: {diff}")
        # Si la diferencia es menor que la tolerancia se corta
        if diff < tol:
            print(f"Convergencia alcanzada en la iteración {i+1}")
            break

        prev_u = u.copy()  # Actualizar la imagen anterior para la siguiente iteración
    
    return u



def descenso_gradiente_momentum(u, f, lambda_param, alpha, beta, num_iter, tol=1e-6):
    v = np.zeros_like(u)  
    prev_u = u.copy()      
    for i in range(num_iter):
        gradiente = grad_objetivo(u, f, lambda_param)
        v = beta * v + (1 - beta) * gradiente  # Actualización del momentum
        u = u - alpha * v  
        diff = np.linalg.norm(u - prev_u) 
        # Imprimir la iteración actual
        print(f"Iteración {i+1}, Diferencia: {diff}")
       # Si la diferencia es menor que la tolerancia se corta
        if diff < tol:
            print(f"Convergencia alcanzada en la iteración {i+1}")
            break
        prev_u = u.copy()  # Actualizar la imagen anterior para la siguiente iteración
    
    return u


def descenso_gradiente_nesterov(u, f, lambda_param, alpha, beta, num_iter, tol=1e-6):
    v = np.zeros_like(u)  
    prev_u = u.copy()  
    for i in range(num_iter):
        u_temp = u - beta * v 
        gradiente = grad_objetivo(u_temp, f, lambda_param)
        v = beta * v + (1 - beta) * gradiente  
        u = u - alpha * v  
        diff = np.linalg.norm(u - prev_u)  
        print(f"Iteración {i+1}, Diferencia: {diff}")

        if diff < tol:
            print(f"Convergencia alcanzada en la iteración {i+1}")
            break
        prev_u = u.copy()  
    
    return u
