------------------------------------------- Loading Keras Model ----------------------------------------------

from keras.models import model_from_json
from keras.models import load_model

json_file = open('model_org.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
net = model_from_json(loaded_model_json)
net.load_weights("model_org.h5")

//net is the final loaded model
