import json

from Class.argv_class import Argv
from Variable.request_variable import REQUEST_BASE, REQUEST_BASE_WITH_URL

argv = Argv()

with open(argv.get_setting(), "r") as f:
    setting = json.load(f)


def model_function_setting():
    output_setting = dict()

    if "url" in setting["DB_API"] and setting["DB_API"]["url"]:
        output_setting["TRAININFO"] = REQUEST_BASE_WITH_URL.format(base_url=setting["DB_API"]["url"], url="/api/train/train_info")
        output_setting["SHARE"] = REQUEST_BASE_WITH_URL.format(base_url=setting["DB_API"]["url"], url="/api/train/shares")
        output_setting["TRAINDATA"] = REQUEST_BASE_WITH_URL.format(base_url=setting["DB_API"]["url"], url="/api/train/train_data")
    else:
        output_setting["TRAININFO"] = REQUEST_BASE.format(ip=setting["DB_API"]["ip"], port=setting["DB_API"]["port"], url="/api/train/train_info")
        output_setting["SHARE"] = REQUEST_BASE.format(ip=setting["DB_API"]["ip"], port=setting["DB_API"]["port"], url="/api/train/shares")
        output_setting["TRAINDATA"] = REQUEST_BASE.format(ip=setting["DB_API"]["ip"], port=setting["DB_API"]["port"], url="/api/train/train_data")

    return output_setting


def kor2eng_function_setting():
    output_setting = dict()

    if "url" in setting["DB_API"] and setting["DB_API"]["url"]:
        output_setting["KOR2ENG_KOR"] = REQUEST_BASE_WITH_URL.format(base_url=setting["DB_API"]["url"], url="/api/train/kor2eng")
    else:
        output_setting["KOR2ENG_KOR"] = REQUEST_BASE.format(ip=setting["DB_API"]["ip"], port=setting["DB_API"]["port"], url="/api/train/kor2eng")

    return output_setting


def tokenizing_setting():
    output_setting = dict()

    if "url" in setting["TOKENIZER_API"] and setting["TOKENIZER_API"]["url"]:
        output_setting["API_POS"] = REQUEST_BASE.format(base_url=setting["TOKENIZER_API"]["url"], url="/api/pos")
    else:
        output_setting["API_POS"] = REQUEST_BASE.format(ip=setting["TOKENIZER_API"]["ip"], port=setting["TOKENIZER_API"]["port"], url="/api/pos")

    return output_setting
