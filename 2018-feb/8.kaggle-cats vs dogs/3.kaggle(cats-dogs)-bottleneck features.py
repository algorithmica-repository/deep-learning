import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

img_width, img_height = 150, 150
nb_train_samples = 2000
nb_validation_samples = 1000
nb_test_samples = 12500
epochs = 30
batch_size = 20

def save_bottlebeck_features(train_dir, validation_dir, test_dir):
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    train_generator = datagen.flow_from_directory(
         train_dir,
         target_size=(img_width, img_height),
         batch_size=batch_size,
         class_mode=None,
         shuffle=False)
    bottleneck_features_train = model.predict_generator(
         train_generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'wb'),
             bottleneck_features_train)
 
    validation_generator = datagen.flow_from_directory(
         validation_dir,
         target_size=(img_width, img_height),
         batch_size=batch_size,
         class_mode=None,
         shuffle=False)
    bottleneck_features_validation = model.predict_generator(
         validation_generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
             bottleneck_features_validation)
    
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_test = model.predict_generator(
        test_generator, nb_test_samples // batch_size)
    np.save(open('bottleneck_features_test.npy', 'wb'),
            bottleneck_features_test)


save_bottlebeck_features(train_dir, validation_dir, test_dir)

train_data = np.load(open('bottleneck_features_train.npy','rb'))
train_labels = np.array( 
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

validation_data = np.load(open('bottleneck_features_validation.npy','rb'))
validation_labels = np.array(
        [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')   
save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

history = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              callbacks=[save_weights, early_stopping])

utils.plot_loss_accuracy(history)

test_data = np.load(open('bottleneck_features_test.npy','rb'))
pred = model.predict_proba(test_data)
submissions=pd.DataFrame({"id": list(range(1,test_data.shape[0])),
                         "label": pred[:,0]})
submissions.to_csv("submission.csv", index=False)