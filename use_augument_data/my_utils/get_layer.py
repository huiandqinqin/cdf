import keras.backend
def get_layer_output(model, x ,index):
    layer = keras.backend.function([model.input], [model.layers[index].output])
    return layer([x])[0]
