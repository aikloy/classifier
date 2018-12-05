from Function.InnerFunction.model_function import load_model_info

from Class.eng_charCNN_with_LSTM_model_class import EngCharCNNWithLSTM as CurrentModel


def load_models(models = None, scope=None):
    if models and scope:
        model_info = load_model_info(scope)
        models[scope] = CurrentModel(scope, model_info["setting"])
    else:
        model_info = load_model_info()
        models = {info["scope"]: CurrentModel(info["scope"], info["setting"]) for info in model_info}

    return models


def train_model(scope):
    model_info = load_model_info(scope)
    if model_info:
        model = CurrentModel(model_info["scope"], model_info["setting"], True)
        result = model.run()

        return result
    else:
        return None


def forward_model(model, msg, num=None):
    answer = model.run(msg)
    if num:
        num = min([num, len(answer)])
        answer = answer[:num]

    return answer
