import cv2
import numpy as np
import sklearn.cluster as cluster # type: ignore
import matplotlib.pyplot as plt
import tensorflow as tf # type: ignore   
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.model_selection import train_test_split 


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convierte arrays numpy a listas
    if isinstance(obj, np.integer):
        return int(obj)  # Convierte enteros numpy a enteros de Python
    if isinstance(obj, np.floating):
        return float(obj)  # Convierte flotantes numpy a flotantes de Python
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def getFeatures(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    mask = np.uint8(1*(gray < threshold))
    
    B=(1/255)*np.sum(img[:,:,0]*mask)/np.sum(mask)
    G=(1/255)*np.sum(img[:,:,1]*mask)/np.sum(mask)
    R=(1/255)*np.sum(img[:,:,2]*mask)/np.sum(mask)
    if np.isnan(B):
        B=0
        print("nann")
    if np.isnan(G):
        G=0
        print("nann")
    if np.isnan(R):
        R=0    
        print("nann")
    return [B,G,R]

def main():

    data_dir = './coil-100/'
    object_ids = [1, 4, 5, 7, 8, 9, 11, 16, 25, 36, 49, 59]
    object_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    images = [f'{data_dir}obj{str(obj_id)}__{j}.png' for obj_id in object_ids for j in range(0, 360, 5)]

    dataset_info = []  # Lista para guardar información completa de cada imagen
    features = []
    labels = []

    # Extracción de características
    for i in range(0, len(images), 72):
        for j in range(0,60):
            img_path = images[j + i]
            img = cv2.imread(img_path)
            feature = getFeatures(img)
            features.append(feature)
            labels.append(object_labels[i // 72])
            
            # Guardar información de la imagen en dataset_info
            dataset_info.append({
                'path': img_path,
                'features': feature,
                'label': object_ids[i // 72],  # Etiqueta original
                'kmeans_label': None  # Inicialmente vacío, se llenará después
            })

    # Convertir a numpy arrays
    features = np.array(features)
    labels = np.array(labels)


    # Visualización inicial
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(features[:, 0], 
                         features[:, 1], 
                         features[:, 2],
                         c=labels,
                         cmap='viridis',
                         marker='o')

    ax.set_xlabel('B (Blue)')
    ax.set_ylabel('G (Green)')
    ax.set_zlabel('R (Red)')
    ax.set_title('RGB Features in 3D Space')
    plt.colorbar(scatter, label='Class')
    plt.show()

    print("features", features.shape)
    print("labels", labels.shape)        

    for i in range(0, len(features), 21):
        print(f'Label: {labels[i]}')

    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    
    print("X_train", X_train.shape)
    for i in range(0, len(X_train), 21):
        print(f'Label: {y_train[i]}')


    y_trainOneHot = tf.keras.utils.to_categorical(y_train, num_classes=12)
    y_testOneHot = tf.keras.utils.to_categorical(y_test, num_classes=12)
    
    
    print("y_trainOneHot", y_trainOneHot.shape)
    for i in range(0, len(X_train), 21):
        print(f'Label: {y_trainOneHot[i]}')


    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_initializer='normal') )
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(12, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())    
    
    model.fit(X_train, y_trainOneHot, epochs=50, batch_size=3)

    loss, accuracy = model.evaluate(X_test, y_testOneHot)

    print(f'Loss: {loss}, Accuracy: {accuracy}')

    #save model en formato keras
    model.save('model.keras')

    print("Modelo guardado")

if __name__ == '__main__':
    main()