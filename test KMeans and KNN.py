import cv2
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

images = load_images_from_folder("../archive/train/train")

def extract_color_histograms(images):
    histograms = []
    for img in images:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_img], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        histograms.append(hist.flatten())
    return histograms

data = extract_color_histograms(images)

from sklearn.cluster import KMeans

num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters,n_init='auto')
kmeans.fit(data)


labels_pred = kmeans.labels_

print(labels_pred)

