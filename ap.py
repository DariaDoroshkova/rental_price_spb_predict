import pandas
from flask import Flask, request
import joblib
import numpy
MODEL_PATH_1 = 'model_1.pkl'
MODEL_PATH_2 = 'model_2.pkl'

ap = Flask(__name__)


@ap.route("/predict_price", methods = ['GET'])
def predict():
    args = request.args
    floor = args.get('floor', default=-1, type=int)
    area = args.get('area', default=-1, type=float)
    kitchen_area = args.get('kitchen_area', default=-1, type=float)
    renovation = args.get('renovation', default=-1, type=int)
    agent_fee = args.get('agent_fee', default=-1, type=float)
    rooms = args.get('rooms', default=-1, type=int)
    model = args.get('model', default = 0, type=int)

    if model == 1:

        model_1 = joblib.load(MODEL_PATH_1)
        x = numpy.array([floor, area, kitchen_area, rooms]).reshape(1,-1)
        result = model_1.predict(x)
        result = result.reshape(1,-1)

        return str(result[0][0])

    if model == 2:
        model_2 = joblib.load(MODEL_PATH_2)
        x_2 = numpy.array([floor, area, kitchen_area, renovation, agent_fee, rooms]).reshape(1, -1)
        result_2 = model_2.predict(x_2)
        result_2 = result_2.reshape(1, -1)

        return str(result_2[0][0])

    else:
        return "model type should be 1 or 2"


if __name__ == '__main__':
    ap.run(debug=True, port=5444, host='0.0.0.0')
