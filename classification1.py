import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(tf.__version__)

states = {"ANDAMAN & NICOBAR ISLANDS":0, "ARUNACHAL PRADESH":1, "ASSAM & MEGHALAYA":2, "NAGA MANI MIZO TRIPURA":3,
          "SUB HIMALAYAN WEST BENGAL & SIKKIM":4, "GANGETIC WEST BENGAL":5, "ORISSA":6, "JHARKHAND":7,
          "BIHAR":8, "EAST UTTAR PRADESH":9, "WEST UTTAR PRADESH":10, "UTTARAKHAND":11, "HARYANA, DELHI AND CHANDIGARH":12,
          "PUNJAB":13, "HIMACHAL PRADESH":14, "JAMMU & KASHMIR":15, "WEST RAJASTHAN":16, "EAST RAJASTHAN":17,
          "WEST MADHYA PRADESH":18, "EAST MADHYA PRADESH":19, "GUJARAT":20, "SAURASHTRA & KUTCH":21,
          "KONKAN & GOA":22, "MADHYA MAHARASHTRA":23, "MATATHWADA":24, "VIDARBHA":25, "CHATTISGARH":26,
          "COASTAL ANDHRA PRADESH":27, "TELENGANA":28, "RAYALSEEMA":29, "TAMIL NADU":30,
          "COASTAL KARNATKA":28, "NORTH INTERIOR KARNATKA":29, "SOUTH INTERIOR KARNATKA":30,
          "KERALA":31, "LAKSHADWEEP":32}


data = [2763.2, 2596.8, 2487, 2363.9, 2908, 1572.1, 1097.2, 1191.5, 1207, 946.8, 771.4, 1712.9, 384.4,
        405.3, 1068.4, 984.6, 235.3, 440, 787, 874.2, 550.6, 286.1, 3133.2, 701.6, 684.4, 1105.2,
        906.3, 992.3, 1078, 871.3, 972, 3351.3, 712.4, 1207.2, 2412.6, 1372.1
        ]

train_data = np.array([data])
# 1 -> less rainfall
# 0 -> heavy rainfall
labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 0, 1, 1, 0, 1]


train_labels = np.array([labels])

# dataset = pd.read_csv("classification.csv")
# # print(dataset)
#
# train_data = dataset.ANNUAL
# # print(train_data)
#
# train_labels = dataset.LABELS
# # print(train_labels)


class_names = ['heavy', 'not-heavy']
# 36
# print(len(annual))
# print(len(labels))
# print("************************")
# print(train_data.shape)
# print(train_labels.shape)
# print("************************")


model = keras.Sequential([
    # keras.layers.Flatten(input_shape=(1, 36)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2,  activation=tf.nn.softmax)

])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_data, train_labels, epochs=2)

