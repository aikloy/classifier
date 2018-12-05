import requests

from Function.load_setting_function import model_function_setting
from Variable.model_variable import CLASSIFIER_CATEGORY
from Variable.request_variable import RESPONSE_OK, RESPONSE_OK_WITH_NO_CONTENT

setting = model_function_setting()


def load_model_info(model_scope=None):
    if model_scope:
        model_info = requests.get("/".join([setting["TRAININFO"], model_scope])).json()
    else:
        model_info = list()
        model_list = [x for x in requests.get(setting["TRAININFO"]).json()]
        for info in model_list:
            model_type = info["model_type"].strip().split("/")
            if model_type[0] == CLASSIFIER_CATEGORY:
                model_info.append(info)

    return model_info


def load_share(scope):
    share = requests.get("/".join([setting["SHARE"], scope]))
    if share.status_code == RESPONSE_OK_WITH_NO_CONTENT or len(share.content) == 0:
        return None
    else:
        share = share.json()
        return share["share"]


def save_share(scope, share):
    prev_share = requests.get("/".join([setting["SHARE"], scope]))
    if prev_share == RESPONSE_OK and len(prev_share.content) > 0:
        prev_share = prev_share.json()
        requests.put("/".join([setting["SHARE"], prev_share[0]["id"]]), json={"scope": scope, "share": share})
    else:
        requests.post(setting["SHARE"], json={"scope": scope, "share": share})


def load_train_data(scope):
    if scope == "카테고리추천":
        train_data = requests.get("/".join([setting["TRAINDATA"], scope]))
        if train_data.status_code == RESPONSE_OK_WITH_NO_CONTENT or len(train_data.content) == 0:
            return None
        else:
            train_data = train_data.json()
            input_data, output_data = list(), list()
            for data in train_data:
                question = data["data"]["input"]
                input_data.append(question)
                output_data.append(data["data"]["output"])

            return input_data, output_data

    elif scope == "태그추천":
        train_data = requests.get("/".join([setting["TRAINDATA"], scope]))
        if train_data.status_code == RESPONSE_OK_WITH_NO_CONTENT or len(train_data.content) == 0:
            return None
        else:
            train_data = train_data.json()
            input_data, output_data = list(), list()
            for data in train_data:
                question = data["data"]["input"]
                input_data.append(question)
                output_data.append(data["data"]["output"])

            return input_data, output_data
    else:
        return None
