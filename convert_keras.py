import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tf_keras.models import load_model

def convert_models():
    print("Programmatically cloning legacy models to bypass Keras 3 from_config() strictness...")

    models_to_convert = [
        ('malaria.h5', (36, 36, 3), 2), 
        ('pneumonia.h5', (36, 36, 1), 2)
    ]
    
    for model_name, shape, classes in models_to_convert:
        old_path = f'models/{model_name}'
        keras_path = f'models/{model_name.replace(".h5", ".keras")}'
        
        print(f"\nProcessing {model_name}...")
        try:
            # Load legacy
            legacy_model = load_model(old_path)
            
            # The error states that we can't reliably load Keras 2 Conv2D json configurations into Keras 3 objects explicitly
            # Instead of relying on from_config, we manually recreate the exact functional architecture we know exists in that model!
            inputs = tf.keras.Input(shape=shape)
            
            # Conv block 1
            x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            
            # Conv block 2
            x = tf.keras.layers.Conv2D(128 if 'pneumonia' in model_name else 256, (3, 3), activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            
            # Conv block 3
            x = tf.keras.layers.Conv2D(256 if 'pneumonia' in model_name else 512, (3, 3), activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(256 if 'pneumonia' in model_name else 96, activation='relu')(x)
            outputs = tf.keras.layers.Dense(classes, activation='softmax')(x)
            
            new_model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Copy all weights sequentially from identical layers
            # Note: Input layers don't have weights, so legacy skip 0 is necessary if legacy is sequential
            weight_idx = 0
            for layer in new_model.layers:
                if layer.weights: # if it has weights to be set
                    while not legacy_model.layers[weight_idx].weights:
                        weight_idx += 1
                    layer.set_weights(legacy_model.layers[weight_idx].get_weights())
                    weight_idx += 1
                
            print("Successfully explicitly cloned architecture and transferred weights.")

            # Save in Keras 3 using tf.keras natively
            new_model.save(keras_path)
            print(f"✅ Successfully converted to {keras_path}")
            
        except Exception as e:
            print(f"❌ Error converting {model_name}: {e}")

if __name__ == '__main__':
    convert_models()
