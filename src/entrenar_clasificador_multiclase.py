import tensorflow as tf
from keras.layers import Input, Dense, GlobalAveragePooling2D, GaussianNoise, Lambda
from keras.layers import RandomFlip, RandomRotation, RandomZoom
from keras.applications import ResNet50
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.applications.resnet50 import preprocess_input
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import cv2

# --- PARÁMETROS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'dataset_multiclase')
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, 'modelos_tf')
MODEL_NAME = 'modelo_clasificacion_frutas.keras'
CLASSES_NAME = 'clases_frutas.txt'

# Hiperparámetros
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.009

def create_model(num_classes):
    """
    Crea el modelo de aumento de datos y el clasificador ResNet50.
    """
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomFlip("vertical"),
        RandomRotation(factor=0.35),
        RandomZoom(height_factor=0.2),
        GaussianNoise(stddev = 0.35) #Las condiciones reales son distintas a las ideales, con una cámara de laptop existe mucho ruido ya que es de baja calidad, asi que este ruido gaussiano ayuda a la precisión en ese entorno
    ], name='data_augmentation')
    x = data_augmentation(inputs)

    x = preprocess_input(x)
    
    base_model = ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

# --- FUNCIONES PARA EL PIPELINE DE DATOS CON CLAHE ---
def apply_clahe(image):
    image_np = image.numpy().astype(np.uint8)
    lab_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_l_channel = clahe.apply(l_channel)
    merged_lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))
    final_image = cv2.cvtColor(merged_lab_image, cv2.COLOR_LAB2RGB)
    return final_image.astype(np.float32)

@tf.function
def tf_apply_clahe(image, label):
    [clahe_image] = tf.py_function(apply_clahe, [image], [tf.float32])
    clahe_image.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return clahe_image, label

def get_label(file_path, class_names):
    parts = tf.strings.split(file_path, os.path.sep)
    return tf.argmax(parts[-2] == class_names)

def decode_img(img):
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    return img

def process_path(file_path, class_names):
    label = get_label(file_path, class_names)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], preserve_aspect_ratio=True)
    img = tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE)
    return img, label

def lr_scheduler(epoch, lr):
    if (epoch + 1) % 7 == 0: return lr * 0.1
    return lr

# --- FUNCIÓN DE TRADUCCIÓN PARA LOS REPORTES ---
def traducir_nombres_clases(nombres_en_ingles):
    diccionario_frutas = {'Apple': 'Manzana', 'Banana': 'Plátano', 'Mango': 'Mango', 'Orange': 'Naranja', 'Strawberry': 'Fresa'}
    diccionario_estados = {'Raw': 'Inmadura', 'Ripe': 'Madura'}
    
    nombres_traducidos = []
    for nombre in nombres_en_ingles:
        try:
            fruta_en, estado_en = nombre.split('_')
            fruta_es = diccionario_frutas.get(fruta_en, fruta_en)
            estado_es = diccionario_estados.get(estado_en, estado_en)
            nombres_traducidos.append(f"{fruta_es} {estado_es}")
        except ValueError:
            nombres_traducidos.append(nombre)
            
    return nombres_traducidos

if __name__ == '__main__':
    AUTOTUNE = tf.data.AUTOTUNE
    
    # --- CONSTRUCCIÓN MANUAL DEL DATASET ---
    train_dir = os.path.join(DATA_DIR, 'train')
    class_names = np.array(sorted([item for item in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, item))]))
    num_classes = len(class_names)
    print(f"Clases encontradas ({num_classes}): {class_names}")

    list_ds_train = tf.data.Dataset.list_files(os.path.join(train_dir, '*/*'), shuffle=True)
    list_ds_test = tf.data.Dataset.list_files(os.path.join(DATA_DIR, 'test', '*/*'), shuffle=False)

    train_dataset = list_ds_train.map(lambda x: process_path(x, class_names), num_parallel_calls=AUTOTUNE)
    test_dataset = list_ds_test.map(lambda x: process_path(x, class_names), num_parallel_calls=AUTOTUNE)

    train_dataset = train_dataset.map(tf_apply_clahe, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.map(tf_apply_clahe, num_parallel_calls=AUTOTUNE)

    train_dataset = train_dataset.batch(BATCH_SIZE).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).cache().prefetch(buffer_size=AUTOTUNE)

    print("\nConfigurando el modelo ResNet-50...")
    model = create_model(num_classes)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    lr_schedule_callback = LearningRateScheduler(lr_scheduler)

    print("\nIniciando entrenamiento...")
    start_time = time.time()
    history = model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset,
                      callbacks=[checkpoint, lr_schedule_callback])
    time_elapsed = time.time() - start_time
    print(f'\nEntrenamiento completado en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    if os.path.exists(model_path):
        print("\nCargando el mejor modelo guardado...")
        model.load_weights(model_path)
        if history.history.get('val_accuracy'):
            best_acc = max(history.history['val_accuracy'])
            print(f'Mejor Precisión en Validación: {best_acc:.4f}')
    else:
        print("\nNo se guardó ningún modelo.")

    classes_path = os.path.join(MODEL_SAVE_DIR, CLASSES_NAME)
    with open(classes_path, "w", encoding='utf-8') as f:
        f.write("\n".join(class_names))
    print(f"Nombres de clases guardados en: {classes_path}")

    # --- EVALUACIÓN DETALLADA DEL MODELO ---
    if os.path.exists(model_path):
        print("\n--- Iniciando Evaluación Detallada del Mejor Modelo ---")
        y_true = []
        test_labels_ds = test_dataset.map(lambda x, y: y)
        for labels_batch in test_labels_ds:
            y_true.extend(labels_batch.numpy())
        
        predictions = model.predict(test_dataset)
        y_pred = np.argmax(predictions, axis=1)
        
        # --- LÓGICA DE TRADUCCIÓN ---
        print("\nTraduciendo nombres de clases para los reportes...")
        class_names_es = traducir_nombres_clases(class_names)
        
        print("\n--- Reporte de Clasificación ---")
        print(classification_report(y_true, y_pred, target_names=class_names_es))
        
        print("\n--- Generando Matriz de Confusión ---")
        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names_es, yticklabels=class_names_es)
        
        plt.title('Matriz de Confusión del Mejor Modelo')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
        confusion_matrix_path = os.path.join(MODEL_SAVE_DIR, 'matriz_de_confusion.png')
        plt.savefig(confusion_matrix_path)
        print(f"\nMatriz de confusión guardada en: {confusion_matrix_path}")
        plt.show()