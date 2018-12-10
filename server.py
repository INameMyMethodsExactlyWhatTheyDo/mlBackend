from keras.models import load_model
import numpy as np
from flask import Flask, Response, request
import tensorflow as tf
import keras
app = Flask(__name__)


def normalize(inputString):
    input = list(map(int, inputString.split()))
    point = input[0]
    output = []
    for x in input:
        output.append((x - point)/point)
    return point, output

def denormalize(array, point):
    print("THIS IS ARRAY " + str(list(array)))
    out = str()
    point = float(point)
    for x in array:
        string = (x * point)
        print("thius is resut " + str(string))
        out += str(string) + " "
    return out


def arrayToInput(array):
    test = np.array(array)
    test = test.reshape(1, test.shape[0], 1)
    return test


@app.route("/", methods=['GET'])
def root():
    return "root"

@app.route('/send/', methods=['POST'])
def get_data():
    print("THIS IS start!\n\n\n\n\n")
    input = str(request.data.decode("utf-8"))
    point, input = normalize(input)
    model = load_model("My_Model.h5")
    
    print("HELLO")

    print("THIS IS POINT " + str(point))
    output = ""
    for x in range(30):
        temp = arrayToInput(input)
        ans = model.predict(temp)
        input.append(ans[0][0])
        output += str(ans[0][0] * point + point) + " "
    keras.backend.clear_session()
    return output

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=5000)
