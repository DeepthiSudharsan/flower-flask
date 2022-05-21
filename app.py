from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import PIL
import tensorflow as tf
import numpy as np
import urllib.request
import os.path
import json
from flask import Flask, request
import base64
import io
app = Flask(__name__)

@app.route('/')
def intro():
    return '<h1>Flower Classification</h1>'

# Use POST to send and receive
@app.route('/mobilenet/predict', methods=['POST'])
def predict_flower():
    data = request.get_json()
    if "myimg" not in data.keys():
        return { "The image is of the flower : " : "IMAGE NOT AVAILABLE" }
    else:
#         with open(json["myimg"]) as jsonfile:
        json_data = data["myimg"]
        # image = Image.fromarray(np.array(json_data, dtype='uint8'))
        img_byte_arr = base64.b64decode(json_data)
        image = Image.open(io.BytesIO(img_byte_arr))
        model_filename = "MobileNet_model.hdf5"
        model = tf.keras.models.load_model(model_filename)
        return predict_image(model, image)
        

def predict_image(model, image):
    lanel_dic = {"Toad lily" :  38, "Love in the mist":  61, "Monkshood":  75, "Azalea":  54, "Fritillary":  6, "Silverbush":  17, "Canterbury bells":  8, "Stemless gentian":  59, "Pink primrose":  103, "Buttercup":  62, "Poinsettia":  92, "Desert-rose":  76, "Bird of paradise":  28, "Columbine":  16, "Cyclamen":  83, "Frangipani":  93, "Sweet pea":  19, "Siam tulip":  26, "Great masterwort":  89, "Hard-leaved pocket orchid":  22, "Marigold":  53, "Foxglove":  57, "Wild pansy":  9, "Windflower":  84, "Daisy":  64, "Tiger lily":  18, "Purple coneflower":  23, "Orange dahlia":  41, "Globe-flower":  43, "Lilac hibiscus":  85, "Fire lily":  3, "Balloon flower":  87, "Iris":  101, "Bishop of llandaff":  71, "Yellow iris":  51, "Garden phlox":  0, "Alpine sea holly":  21, "Geranium":  60, "Pink quill":  35, "Tree poppy":  44, "Spear thistle":  69, "Bromelia":  82, "Common dandelion":  50, "Sword lily":  97, "Peruvian lily":  91, "Carnation":  96, "Cosmos":  46, "Spring crocus":  25, "Lotus":  94, "Bolero deep blue":  74, "Anthurium":  79, "Rose":  96, "Water lily":  32, "Primula":  5, "Blackberry lily":  70, "Gaura":  95, "Trumpet creeper":  52, "Globe thistle":  7, "Sweet william":  40, "Hippeastrum":  15, "Snapdragon":  47, "Mexican petunia":  49, "Petunia":  15, "Gazania":  10, "King protea":  11, "Blanket flower":  34, "Common tulip":  102, "Giant white arum lily":  65, "Wild rose":  1, "Morning glory":  4, "Thorn apple":  98, "Pincushion flower":  39, "Tree mallow":  13, "Canna lily":  91, "Camellia":  99, "Pink-yellow dahlia":  63, "Bee balm":  80, "Wild geranium":  24, "Artichoke":  38, "Black-eyed susan":  58, "Ruby-lipped cattleya":  86, "Clematis":  55, "Prince of wales feathers":  81, "Hibiscus":  42, "Cautleya spicata":  67, "Lenten rose":  36, "Red ginger":  14, "Colt's foot":  90, "Mallow":  31, "Californian poppy":  68, "Corn poppy":  52, "Moon orchid":  45, "Passion flower":  48, "Grape hyacinth":  78, "Japanese anemone":  66, "Watercress":  72, "Cape flower":  29, "Osteospermum":  77, "Barberton daisy":  20, "Bougainvillea":  27, "Magnolia":  100, "Sunflower":  90, "Daffodil":  12, "Wallflower":  56}
#     img_filename = "0a4ddf5c2.jpeg"

    X_val = []
#         image = Image.open(img_filename)     
    image = np.array(image)
    image_data_as_arr = np.asarray(image)
    X_val.append(image_data_as_arr)
    X_val = np.asarray(X_val)   
    X_val = tf.expand_dims(X_val, axis=-1)
   
    y_pred = model.predict(X_val)
    Y_pred_classes = np.argmax(y_pred,axis=1)
    keys = [k for k, v in lanel_dic.items() if v == Y_pred_classes]
    return  { "The image is of the flower : " : keys[0] }

if __name__ == "__main__":
    app.run(debug=True)
    # # load from file
    # img_filename = "0a4ddf5c2.jpeg"
    # image = Image.open(img_filename)

    # # Convert to base64
    # img_byte_arr = io.BytesIO()
    # image.save(img_byte_arr, format='PNG')
    # img_byte_arr = img_byte_arr.getvalue()
    # data = base64.b64encode(img_byte_arr)
    # with open("sample.json", "w") as outfile:
    #     outfile.write(data.decode("utf-8")) 
    # # convert back to image
    # img_byte_arr = base64.b64decode(data)
    # image = Image.open(io.BytesIO(img_byte_arr))

    # # load a model
    # model_filename = "MobileNet_model.hdf5"
    # model = tf.keras.models.load_model(model_filename)
    # print(predict_image(model, image))