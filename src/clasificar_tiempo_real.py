import keras
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

def run_realtime_classifier():
    MODEL_SAVE_DIR = 'modelos_tf'
    MODEL_NAME = 'modelo_clasificacion_frutas.keras'
    CLASSES_NAME = 'clases_frutas.txt'
    MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    CLASSES_PATH = os.path.join(MODEL_SAVE_DIR, CLASSES_NAME)
    FONT_PATH = "DejaVuSans.ttf"
    
    # Diccionario de traducción de frutas
    traduccion_frutas = {'Apple': 'Manzana', 'Banana': 'Plátano', 'Mango': 'Mango', 'Orange': 'Naranja', 'Strawberry': 'Fresa'}
    traduccion_estados = {'Raw': 'Inmaduro', 'Ripe': 'Maduro'}

    try:
        with open(CLASSES_PATH) as f: class_names = [line.strip() for line in f.readlines()]
    except FileNotFoundError: print(f"Error: No se encontró '{CLASSES_PATH}'."); return
    
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("Modelo cargado y listo para evaluación.")
    except Exception as e:
        print(f"Error: No se pudo cargar el modelo desde '{MODEL_PATH}'. Error: {e}"); return

    try:
        font_label = ImageFont.truetype(FONT_PATH, 24)
        font_conf = ImageFont.truetype(FONT_PATH, 18)
    except IOError:
        print(f"Fuente no encontrada en '{FONT_PATH}'. Usando fuente por defecto.")
        font_label = ImageFont.load_default()
        font_conf = ImageFont.load_default()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Error: No se puede abrir la cámara."); return

    while True:
        ret, frame_cv = cap.read()
        if not ret: break
        frame_cv = cv2.flip(frame_cv, 1)

        pil_image = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
        img_resized = pil_image.resize((224, 224))
        img_array = np.array(img_resized)
        input_batch = np.expand_dims(img_array, axis=0)

        predictions = model.predict(input_batch)
        confidence_score = np.max(predictions[0]) * 100
        predicted_idx = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_idx]

        CONFIDENCE_THRESHOLD = 50.0 
        pil_draw_image = Image.fromarray(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_draw_image)

        if confidence_score > CONFIDENCE_THRESHOLD:
            try:
                fruta_en, estado_en = predicted_class_name.split('_')
                fruta_es = traduccion_frutas.get(fruta_en, fruta_en)
                estado_es = traduccion_estados.get(estado_en, estado_en)
                
                if fruta_es is None:
                    raise ValueError("Fruta no encontrada en el diccionario")

                color = (0, 255, 0) if estado_en == 'Ripe' else (0, 165, 255)
                h, w, _ = frame_cv.shape
                x1, y1, x2, y2 = (w - 300) // 2, (h - 300) // 2, (w + 300) // 2, (h + 300) // 2
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                label = f'{fruta_es} ({estado_es})'
                conf_text = f'{confidence_score:.1f}%'
                draw.text((x1 + 5, y1 - 30), label, font=font_label, fill=color)
                draw.text((x1 + 5, y2 + 10), conf_text, font=font_conf, fill=color)
            except (ValueError, AttributeError):
                pass
        else:
            draw.text((20, 40), 'Acerque una fruta al centro', font=font_label, fill=(255, 255, 255))
        
        frame_to_show = cv2.cvtColor(np.array(pil_draw_image), cv2.COLOR_RGB2BGR)
        cv2.imshow('Clasificador de Frutas - Presione "q" para salir', frame_to_show)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_realtime_classifier()