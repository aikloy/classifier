import os
import argparse
from flask import Flask, request
from flask_restplus import Resource, Api, fields

from Class.argv_class import Argv
from Function.InnerFunction.get_request_function import get_request
from Variable.request_variable import RESPONSE_OK, RESPONSE_INVALID_INPUT,\
    MODEL_NOT_EXIST, TRAIN_DATA_NOT_EXIST,\
    TRAIN, TEST, FORWARD

parser = argparse.ArgumentParser()
parser.add_argument("--setting_filename", type=str, action="store", required=False, default="setting")

args = parser.parse_args()
current_dir_path = os.path.dirname(os.path.realpath(__file__))

argv = Argv()
argv.set_setting(os.path.join(current_dir_path, "setting", args.setting_filename + ".json"))

from Function.control_function import load_models, train_model, forward_model


application = Flask(__name__)
api = Api(application, default="Classifier Model control", doc="/Classifier/Swagger/", title="Classifier Model control", description="Classifier Model control API")

models = load_models()


@api.route("/Classifier/train")
class Train(Resource):
    @staticmethod
    @api.expect(api.model("train", {
        "scope": fields.String
    }))
    def post():
        global models

        request_body = get_request(request)
        if "scope" in request_body and isinstance(request_body["scope"], str):
            result = train_model(request_body["scope"])
            if result is None:
                return MODEL_NOT_EXIST, RESPONSE_INVALID_INPUT
            elif result is False:
                return TRAIN_DATA_NOT_EXIST, RESPONSE_INVALID_INPUT
            else:
                models = load_models(models, request_body["scope"])

                return result, RESPONSE_OK
        else:
            return TRAIN, RESPONSE_OK

# @api.route("/Classifier/EngCharCNNWithLSTM/test")
# class ClassifierTest(Resource):
#     @staticmethod
#     def post():
#         request_body = json.loads(str(request.data, "utf-8"))
#         if "setting" in request_body and "data" in request_body:
#             model = EngCharCNNWithLSTM(scope="test")
#             answer = model.test(request_body["setting"], request_body["data"])
#
#             return answer, 200
#         else:
#             return {"error": "Invalid input", "required": ["scope", "setting", "data"]}, 400


@api.route("/Classifier/forward")
class Forward(Resource):
    @staticmethod
    @api.expect(api.model("test", {
        "scope": fields.String,
        "msg": fields.String
    }))
    def post():
        request_body = get_request(request)
        if "scope" in request_body and isinstance(request_body["scope"], str) and "msg" in request_body and isinstance(request_body["msg"], str):
            if "limit" in request_body and isinstance(request_body["limit"], int):
                if request_body["scope"] in models:
                    model = models[request_body["scope"]]
                    answer = forward_model(model, request_body["msg"], request_body["limit"])

                    return answer, RESPONSE_OK
                else:
                    return MODEL_NOT_EXIST, RESPONSE_INVALID_INPUT
            else:
                if request_body["scope"] in models:
                    model = models[request_body["scope"]]
                    answer = forward_model(model, request_body["msg"])

                    return answer, RESPONSE_OK
                else:
                    return MODEL_NOT_EXIST, RESPONSE_INVALID_INPUT
        else:
            return FORWARD, RESPONSE_INVALID_INPUT


if __name__ == "__main__":
    application.run("0.0.0.0", 12004)
