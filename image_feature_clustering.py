import cv2
import numpy as np
import sklearn.cluster as cluster # type: ignore
import matplotlib.pyplot as plt   
from sklearn.preprocessing import StandardScaler # type: ignore

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
    
    return [B,G,R]

def main():

    data_dir = './coil-100/'
    object_ids = [1, 4, 5, 7, 8, 9, 11, 16, 25, 36, 49, 59]
    images = [f'{data_dir}obj{str(obj_id)}__{j}.png' for obj_id in object_ids for j in range(0, 360, 5)]

    dataset_info = []  # Lista para guardar información completa de cada imagen
    features = []
    labels = []

    # Extracción de características
    for i in range(0, len(images), 72):
        for j in range(0, 21):
            img_path = images[j + i]
            img = cv2.imread(img_path)
            feature = getFeatures(img)
            features.append(feature)
            labels.append(object_ids[i // 72])
            
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

    # Aplicar KMeans
    kmeans = cluster.KMeans(n_clusters=12)
    kmeans.fit(features)
    labels_K = kmeans.labels_
    centers_K = kmeans.cluster_centers_

    # Actualizar etiquetas de KMeans en dataset_info
    for idx, label in enumerate(labels_K):
        dataset_info[idx]['kmeans_label'] = label

    # Visualización con etiquetas de KMeans
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(features[:, 0], 
                         features[:, 1], 
                         features[:, 2],
                         c=labels_K,
                         cmap='viridis',
                         marker='o')
    ax.scatter(centers_K[:, 0], centers_K[:, 1], centers_K[:, 2], c='red', s=100, alpha=0.5)
    ax.set_xlabel('B (Blue)')
    ax.set_ylabel('G (Green)')
    ax.set_zlabel('R (Red)')
    ax.set_title('RGB Features in 3D Space')
    plt.colorbar(scatter, label='Class')
    plt.show()

    import json

    with open('data.json', 'w') as f:
        json.dump(dataset_info, f, default=convert_to_serializable, indent=4)

if __name__ == '__main__':
    main()