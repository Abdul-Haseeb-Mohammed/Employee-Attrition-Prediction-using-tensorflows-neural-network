import tensorflow as tf

def set_random_seed(seed=42):
    tf.keras.utils.set_random_seed(seed)

def declare_model(layers_config):
    model = tf.keras.Sequential()
    for layer in layers_config:
        neurons = layer.get('neurons', 1)
        activation = layer.get('activation', 'linear')
        model.add(tf.keras.layers.Dense(neurons, activation=activation))
    return model

def compile_model(model, loss='binary_crossentropy', optimizer='SGD', learning_rate=None):
    if learning_rate is not None:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, epochs, callbacks=None):
    history = model.fit(x_train, y_train, epochs=epochs, verbose=0, callbacks=callbacks)
    return model, history

def evaluate_model(model, x_train, y_train):
    loss, accuracy = model.evaluate(x_train, y_train)
    return loss, accuracy

def create_lr_scheduler():
    return tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 0.001 * 0.9**(epoch / 3)
    )
