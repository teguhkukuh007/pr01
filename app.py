from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
from Ml_class import Ml_class



app = Flask(__name__)
api = Api(app)

#parser = reqparse.RequestParser()
#parser.add_argument('username', type=unicode, location='json')
#parser.add_argument('password', type=unicode, location='json')

class GenerateModel(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        datagoal = json_data['goal']
        data = json_data['feature']
        datafeature = json_data['data']

        feature={}

        for ft in data:
            feature[ft]=datafeature[ft]

        goal={}
        goal[datagoal]=datafeature[datagoal]
        
       
        ml = Ml_class(feature,goal)
        ml.modelnaivebayes()
        ml.modellogistic()
        return jsonify(message='success',id=ml.id)

api.add_resource(GenerateModel, '/getModel')

class ModelNaivebayes(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        datagoal = json_data['goal']
        data = json_data['feature']
        model=json_data['idmodel']

        datafeature = json_data['data']

        feature={}

        for ft in data:
            feature[ft]=datafeature[ft]

        goal={}
        goal[datagoal]=datafeature[datagoal]
        
       
        ml = Ml_class(feature,goal)
        ml.setId(model)
        hasil=ml.trainNaivebayes()
        metrik={}
        metrik["precision"]=ml.getPrecision()
        metrik["recall"]=ml.getRecall()
        metrik["f1score"]=ml.getFscore()
        metrik["accuracy"]=ml.getAccuracy()
        
        return jsonify(message='success',metrik=metrik,hasil=hasil.tolist())

api.add_resource(ModelNaivebayes, '/trainNb')

class ModelLogistic(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        datagoal = json_data['goal']
        data = json_data['feature']
        model=json_data['idmodel']

        datafeature = json_data['data']

        feature={}

        for ft in data:
            feature[ft]=datafeature[ft]

        goal={}
        goal[datagoal]=datafeature[datagoal]
        
       
        ml = Ml_class(feature,goal)
        ml.setId(model)
        hasil=ml.trainLogistic()
        metrik={}
        metrik["precision"]=ml.getPrecision()
        metrik["recall"]=ml.getRecall()
        metrik["f1score"]=ml.getFscore()
        metrik["accuracy"]=ml.getAccuracy()
        
        return jsonify(message='success',metrik=metrik,hasil=hasil.tolist())

api.add_resource(ModelLogistic, '/trainLg')

class ModelNn(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        datagoal = json_data['goal']
        data = json_data['feature']
        model=json_data['idmodel']

        datafeature = json_data['data']

        feature={}

        for ft in data:
            feature[ft]=datafeature[ft]

        goal={}
        goal[datagoal]=datafeature[datagoal]
        
       
        ml = Ml_class(feature,goal)
        ml.setId(model)
        hasil=ml.trainNn()
        metrik={}
        metrik["precision"]=ml.getPrecision()
        metrik["recall"]=ml.getRecall()
        metrik["f1score"]=ml.getFscore()
        metrik["accuracy"]=ml.getAccuracy()
        
        return jsonify(message='success',metrik=metrik,hasil=hasil.tolist())

api.add_resource(ModelNn, '/trainNn')

if __name__ == '__main__':
    app.run(debug=True)