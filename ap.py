import pandas
from flask import Flask, request
import joblib
import numpy
MODEL_PATH = 'model.pkl'

ap = Flask(__name__)
model = joblib.load(MODEL_PATH)

@ap.route("/predict_price", methods = ['GET'])
def predict():
    args = request.args
    floor = args.get('floor', default=-1, type=int)
    open_plan = args.get('open_plan', default=-1, type=int)
    rooms = args.get('rooms', default=-1, type=int)
    studio = args.get('studio', default=-1, type=int)
    area = args.get('area', default=-1, type=float)
    kitchen_area = args.get('kitchen_area', default= -1, type=float)

    # response = "open_plan:{}, rooms:{}, area:{}, renovation:{}".format(open_plan, rooms, area, renovation)


    x = numpy.array([floor, open_plan, rooms, studio, area, kitchen_area]).reshape(1,-1)
    result = model.predict(x)
    result = result.reshape(1,-1)

    return str(result[0][0])


if __name__ == '__main__':
    ap.run(debug=True, port=5444, host='0.0.0.0')