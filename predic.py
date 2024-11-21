import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_metrics(accuracy, precision, recall, f1_score):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1_score]

    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
    plt.ylim(0, 1)
    plt.title('Model Performance Metrics')
    plt.ylabel('Value')
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)
    plt.show()

def getFeatures(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    mask = np.uint8(1 * (gray < threshold))
    B = (1/255) * np.sum(img[:,:,0] * mask) / np.sum(mask)
    G = (1/255) * np.sum(img[:,:,1] * mask) / np.sum(mask)
    R = (1/255) * np.sum(img[:,:,2] * mask) / np.sum(mask)
    if np.isnan(B):
        B = 0
    if np.isnan(G):
        G = 0
    if np.isnan(R):
        R = 0
    return [B, G, R]

def main():
    # Cargar el modelo entrenado
    model = tf.keras.models.load_model('./model.h5')
    #model = tf.keras.models.load_model('./model.keras')
    
    # Clasificar una nueva imagen
    data_dir = './coil-100/'
    object_ids = [1, 4, 5, 7, 8, 9, 11, 16, 25, 36, 49, 59]
    object_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    images = [f'{data_dir}obj{str(obj_id)}__{j}.png' for obj_id in object_ids for j in range(0, 360, 5)]
        
    print(len(images))
    
    data_set_info = []
    for i in range(0, len(images)):
        img_path = images[i]
        img = cv2.imread(img_path)
        feature = getFeatures(img)
        label = object_labels[i // 72]
        
        data_set_info.append({
            'path': img_path,
            'features': np.array(feature).reshape(1, -1),
            'label': label,
            'label_pred': None
        })

    print(len(data_set_info))


    for data in data_set_info:
        features = data['features']
        predic = model.predict(features)        
        data['label_pred'] = np.argmax(predic, axis=1) 

#    for data in data_set_info:
#        print(f'Path: {data["path"]}')
#        print(f'Original label: {data["label"]}')
#        print(f'Predicted label: {data["label_pred"]}')
#        print('----------------------------------')

    # Almacenar etiquetas reales y predicciones
    y_true = []
    y_pred = []

    for data in data_set_info:
        y_true.append(data['label'])
        y_pred.append(data['label_pred'][0])  # label_pred es un array, toma el valor

    # Calcular la matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    print("Matriz de Confusión:")
    print(cm)

    # Extraer métricas de la matriz de confusión
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    print("\nValores:")
    print(f"TP: {TP.sum()}, FP: {FP.sum()}, FN: {FN.sum()}, TN: {TN.sum()}")

    # Calcular Accuracy, Recall, Precision y F1-Score
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    precision = np.mean(TP / (TP + FP + 1e-10))  # Evitar divisiones por cero
    recall = np.mean(TP / (TP + FN + 1e-10))
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1_score:.2f}")

    # Opcional: Reporte completo por clase
    report = classification_report(y_true, y_pred, target_names=[f'Class {i}' for i in range(len(object_labels))])
    print("\nReporte Clasificación:")
    print(report)

        # Visualizar matriz de confusión
    class_names = [f'Class {i}' for i in range(len(object_labels))]
    plot_confusion_matrix(cm, class_names)

    # Visualizar métricas
    plot_metrics(accuracy, precision, recall, f1_score)





if __name__ == '__main__':
    main()