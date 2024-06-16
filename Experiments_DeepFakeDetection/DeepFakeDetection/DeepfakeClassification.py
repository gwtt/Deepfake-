
import os
from PIL import Image
import numpy as np
import pickle

directory_path_fake = "./DFDCFake"


fake_image_array = []
def func(folder_path,array):
    count=0
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
        # Check if the file is a valid image file (you can customize this check based on your needs)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Create the full path to the image using os.path.join()
                image_path = os.path.join(root, filename)
                # Open the image using the PIL library
                image = Image.open(image_path)
                try:
                    resized_image = image.resize((128, 128))
                    image_array = np.array(resized_image) / 255
                    array.append(image_array)
                except Exception  as e:
                    print(f"Error opening image: {image_path}")
                    continue
            print("->",count)
            count += 1

func(directory_path_fake,fake_image_array)


fake_image_array = np.array(fake_image_array)

print("训练假图数量",len(fake_image_array))


directory_path_real = "./DFDCReal"

real_image_array=[]

func(directory_path_real,real_image_array)

real_image_array = np.array(real_image_array)

print("训练真图数量",len(real_image_array))


# for i in range(0,len(real_image_array)):
#     if(real_image_array[i].shape ==  (128,128,3) ):
#         k.append(real_image_array[i])



X =[]

for i in range(len(fake_image_array)):
    X.append(fake_image_array[i])
for i in range(len(real_image_array)):
    X.append(real_image_array[i])
X=np.array(X)


y =[]
for i in range(len(fake_image_array)):
    y.append(0)
for i in range(len(real_image_array)):
    y.append(1)
y=np.array(y)
print(len(y))


from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# X_train_resized = np.array([np.array(Image.fromarray(img).resize((128, 128))) for img in X_train])
# X_test_resized = np.array([np.array(Image.fromarray(img).resize((128, 128))) for img in X_test])

# Data augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# Data augmentation using ImageDataGenerator
batch_size =16
train_datagen = datagen.flow(X_train, y_train, batch_size=batch_size)

# Train the simplified model with increased data augmentation
epochs=20
history = model.fit(
    train_datagen,
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test)
)



import matplotlib.pyplot as plt


# Plot accuracy and loss graphs
plt.figure(figsize=(12, 4))

# Plot Training & Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Training & Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model on the testing data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Evaluate the model on the testing data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Generate predictions for the test set
y_pred = model.predict(X_test)
threshhold=0.5
y_pred_classes = (y_pred > threshhold).astype(int)  


model.save('./save/DFDCmodel', save_format='tf')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


cm = confusion_matrix(y_test, y_pred_classes)

# Visualize the confusion matrix with text annotations
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks(ticks=range(2), labels=["0:fake", "1:real"])  
plt.yticks(ticks=range(2), labels=["0:fake", "1:real"])
plt.title("Confusion Matrix")


for i in range(2):
    for j in range(2):
        text = plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=12, color="white")

plt.show()

directory_path_validate = "./TestFake"


validate_array = []


def func(folder_path, array):
    count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert()
            resized_image = image.resize((128, 128))
            image_array = np.array(resized_image)/255

            array.append(image_array)
            count += 1

func(directory_path_validate, validate_array)



validate_array = np.array(validate_array)



def prediction_on_diff_images(validate_array):

    predictions = model.predict(validate_array)
    deepFake=0

    for i in range(len(predictions)):
        if( predictions[i] < 0.5):
            deepFake +=1
    # print("Number of  fake images out of -> ",len(predictions)," is ",deepFake)
    
    return deepFake/len(validate_array),deepFake


a1,deepfake_count=prediction_on_diff_images(validate_array)



print("识别准确率",a1)
print("假图数量：",deepfake_count)


