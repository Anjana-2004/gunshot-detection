from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def get_model(model_type='svm', input_shape=None, num_classes=2):
    """
    Returns a model based on model_type:
    'svm' -> SVC model,
    'rf' -> RandomForestClassifier,
    'cnn' -> Simple CNN (requires input_shape and num_classes)
    """
    if model_type == 'svm':
        return SVC(kernel='rbf', probability=True)

    elif model_type == 'rf':
        return RandomForestClassifier(n_estimators=100)

    elif model_type == 'cnn':
        if input_shape is None:
            raise ValueError("input_shape must be provided for CNN")
        
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')  # change to sigmoid for binary
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
                      metrics=['accuracy'])
        return model

    else:
        raise ValueError("Unknown model_type")

def save_model(model, filename):
    """
    Save SVM/RF using joblib, CNN using Keras .save()
    """
    # Keras models have .save attribute, sklearn models don’t
    if hasattr(model, 'save'):  
        model.save(filename)  # saves full model (architecture + weights)
    else:
        joblib.dump(model, filename)

def load_model(filename):
    """
    Load SVM/RF using joblib, CNN using keras.models.load_model
    """
    try:
        from tensorflow.keras.models import load_model as keras_load
      
        return keras_load(filename)
    except Exception:
        return joblib.load(filename)
