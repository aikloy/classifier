import requests

from Function.load_setting_function import tokenizing_setting
from Variable.tokenize_variable import VALID_TAG

setting = tokenizing_setting()


def get_valid_tag(_sent):
    tokenize_sent = requests.post(setting["API_POS"], json={"sent": _sent}).json()
    valid_sent = [word for word, tag in tokenize_sent if tag in VALID_TAG]

    return valid_sent
