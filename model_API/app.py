from flask import Flask, request
# import canny_edge_detector as ced
from utils import utils
import json
import VGGCAPSNET as vgg

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/CannyEdgeDetection', methods=["GET", "POST"])
def CannyEdgeDetection():
    if request.method == 'POST':
        data = request.get_json()
        # print(data)
        x = data["base64Image"].split(",")
        base64_img = x[1]
        size, img = utils.load_singleImage(base64_img)
        pred, prediction_class = vgg.predict_input(img)
        return_dict = {
            "base64Image": base64_img,
            "size": size,
            "label": str(prediction_class[0]),
            "accuracy": str(pred[0][prediction_class[0]])
        }
        # print("OK")
        # print(return_dict)
    return(json.dumps(return_dict))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)