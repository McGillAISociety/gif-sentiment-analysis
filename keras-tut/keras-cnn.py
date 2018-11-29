from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), strides=(1,1), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size(2,2), strides=(2,2)))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[history])
