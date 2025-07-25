# model_pipeline/models.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout,
    RandomFlip, RandomRotation, RandomZoom, Input
)
# --- ALTERAÇÃO AQUI ---
# Importamos a ResNet50V2 junto com a EfficientNetB0 (ou no lugar dela)
from tensorflow.keras.applications import ResNet50V2

def build_model(num_classes, img_size=(224, 224), dropout_rate=0.4):
    """
    Constrói o modelo de classificação, agora usando ResNet50V2 como base pré-treinada.
    """
    # --- ALTERAÇÃO PRINCIPAL AQUI ---
    # Trocamos a linha que define o modelo base para usar ResNet50V2
    base_model = ResNet50V2(
        input_shape=img_size + (3,), 
        include_top=False,        # Não incluir a camada de classificação original da ImageNet
        weights='imagenet'        # Usar os pesos pré-treinados da ImageNet
    )
    
    # O resto da arquitetura permanece o mesmo
    base_model.trainable = False

    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal"), 
        RandomRotation(0.1), 
        RandomZoom(0.1),
    ], name="data_augmentation")

    inputs = Input(shape=img_size + (3,))
    x = data_augmentation(inputs)
 
    x = base_model(x, training=False)
    
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model, base_model
