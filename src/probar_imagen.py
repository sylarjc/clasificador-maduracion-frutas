import keras
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import ttk, filedialog
import os

# --- CONFIGURACION ---
RUTA_MODELO = "models/modelo_clasificacion_frutas.keras"
RUTA_CLASES = "models/clases_frutas.txt"

# Diccionario para traducir los nombres de clases
DICCIONARIO_TRADUCCION = {
    'Apple': 'Manzana',
    'Banana': 'Platano',
    'Mango': 'Mango',
    'Orange': 'Naranja',
    'Strawberry': 'Fresa',
    'Ripe': 'Maduro',
    'Raw': 'Inmaduro'
}

def traducir_clase(nombre_clase):
    try:
        nombre_fruta_en, estado_en = nombre_clase.split('_')
        nombre_fruta_es = DICCIONARIO_TRADUCCION.get(nombre_fruta_en, nombre_fruta_en)
        estado_es = DICCIONARIO_TRADUCCION.get(estado_en, estado_en)
        return nombre_fruta_es, estado_es
    except ValueError:
        return nombre_clase, "N/A"

def clasificar_imagen(ruta_imagen, modelo, nombres_clases):
    try:
        imagen_pil = Image.open(ruta_imagen).convert('RGB')
        imagen_resized = imagen_pil.resize((224, 224))
        imagen_array = np.array(imagen_resized)
        lote_entrada = np.expand_dims(imagen_array, axis=0)
        predicciones = modelo.predict(lote_entrada)
        puntuacion_confianza = np.max(predicciones[0]) * 100
        indice_predicho = np.argmax(predicciones[0])
        clase_predicha = nombres_clases[indice_predicho]
        fruta_es, estado_es = traducir_clase(clase_predicha)
        imagen_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
        texto_fruta = f"Fruta: {fruta_es}"
        texto_estado = f"Estado: {estado_es}"
        texto_confianza = f"Confianza: {puntuacion_confianza:.2f}%"
        color = (0, 200, 0) if estado_es == 'Maduro' else (0, 100, 255)
        (h, w) = imagen_cv.shape[:2]
        escala_fuente = min(w, h) / 600.0
        grosor_fuente = max(1, int(escala_fuente * 2))
        cv2.putText(imagen_cv, texto_fruta, (15, 40), cv2.FONT_HERSHEY_DUPLEX, escala_fuente, color, grosor_fuente)
        cv2.putText(imagen_cv, texto_estado, (15, 40 + int(50*escala_fuente)), cv2.FONT_HERSHEY_DUPLEX, escala_fuente, color, grosor_fuente)
        cv2.putText(imagen_cv, texto_confianza, (15, 40 + int(100*escala_fuente)), cv2.FONT_HERSHEY_DUPLEX, escala_fuente, color, grosor_fuente)
        detalles_prediccion = {"Fruta": fruta_es, "Estado": estado_es, "Confianza": f"{puntuacion_confianza:.2f}%"}
        return imagen_cv, detalles_prediccion
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None, None

def abrir_archivo_y_clasificar():
    ruta_archivo = filedialog.askopenfilename(
        title="Seleccionar una imagen de fruta",
        filetypes=(("Archivos de imagen", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*"))
    )
    if not ruta_archivo: return
    imagen_clasificada, detalles = clasificar_imagen(ruta_archivo, modelo_cargado, nombres_clases_cargados)
    if imagen_clasificada is not None:
        print("\n--- Resultado de la Clasificación ---")
        for clave, valor in detalles.items():
            print(f"{clave}: {valor}")
        print("-----------------------------------")

        # Para redimensionar la imagen a 640x480 antes de mostrarla
        imagen_redimensionada = cv2.resize(imagen_clasificada, (640, 480))
        cv2.imshow("Resultado de la Clasificacion", imagen_redimensionada)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        with open(RUTA_CLASES, 'r', encoding='utf-8') as f:
            nombres_clases_cargados = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de clases en '{RUTA_CLASES}'."); exit()
    try:
        modelo_cargado = keras.models.load_model(RUTA_MODELO)
        print(f"Modelo '{RUTA_MODELO}' cargado exitosamente.")
    except Exception as e:
        print(f"Error: No se pudo cargar el archivo del modelo desde '{RUTA_MODELO}'. Error: {e}"); exit()

    ventana_principal = tk.Tk()
    ventana_principal.title("Analizador de Madurez de Frutas")
    ventana_principal.geometry("400x200")
    ventana_principal.resizable(False, False)
    estilo = ttk.Style(); estilo.theme_use('clam')
    estilo.configure("TLabel", background="#f0f0f0", foreground="#333", font=("Segoe UI", 12))
    estilo.configure("TButton", font=("Segoe UI", 12, "bold"), padding=10, borderwidth=0, background="#00529B", foreground="white")
    estilo.map("TButton", background=[('active', '#003F7A')], foreground=[('active', 'white')])
    ventana_principal.configure(bg="#f0f0f0")
    marco_principal = ttk.Frame(ventana_principal, padding="20 20 20 20", style="TFrame")
    marco_principal.pack(expand=True, fill="both")
    etiqueta_instruccion = ttk.Label(marco_principal, text="Seleccione una imagen para analizar su tipo y estado de madurez.", wraplength=350, justify="center")
    etiqueta_instruccion.pack(pady=(0, 20))
    boton_abrir = ttk.Button(marco_principal, text="Seleccionar Imagen", command=abrir_archivo_y_clasificar, style="TButton")
    boton_abrir.pack(ipadx=20)
    ventana_principal.mainloop()