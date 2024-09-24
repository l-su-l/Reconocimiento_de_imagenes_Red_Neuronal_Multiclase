import os
import random

# Ruta a la carpeta que contiene las imágenes
folder_path = 'Pack Imagenes de Animales #2 Ultra HD'

# Conjunto de etiquetas
etiquetas = ['horse', 'persona', 'elefante']

# Recorre todos los archivos en la carpeta
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Genera una etiqueta aleatoria
        etiqueta = random.choice(etiquetas)
        
        # Construye el nuevo nombre de archivo
        nombre_archivo, extension = os.path.splitext(filename)
        nuevo_nombre = f"{etiqueta}-{nombre_archivo}{extension}"
        
        # fucion de carpeta con archivos 
        ruta_actual = os.path.join(folder_path, filename)
        ruta_nueva = os.path.join(folder_path, nuevo_nombre)
        
        # Renombra el archivo
        os.rename(ruta_actual, ruta_nueva)
        
        print(f"Renombrado {filename} a {nuevo_nombre}")

        ''' [Salida del print]
        Renombrado 103296.jpg a horse-103296.jpg   
        Renombrado 150464.jpg a elefante-150464.jpg
        Renombrado 188573.jpg a horse-188573.jpg
        Renombrado 20658.jpg a persona-20658.jpg
        Renombrado 262790.jpg a elefante-262790.jpg
        Renombrado 318728.jpg a elefante-318728.jpg
        Renombrado 364184.jpg a persona-364184.jpg
        Renombrado 372048.jpg a persona-372048.jpg
        Renombrado 380979.jpg a elefante-380979.jpg
        Renombrado 405997.jpg a persona-405997.jpg
        Renombrado 500507.jpg a elefante-500507.jpg
        Renombrado 548670.jpg a persona-548670.jpg
        Renombrado 638321.jpg a horse-638321.jpg
        '''