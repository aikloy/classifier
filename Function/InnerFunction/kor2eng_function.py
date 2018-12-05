import hgtk
import requests
from hgtk.exception import NotHangulException

from Function.load_setting_function import kor2eng_function_setting

setting = kor2eng_function_setting()


def kor2eng(word):
    new_word = ""
    for char in word:
        try:
            splite_char = hgtk.letter.decompose(char)
            for component in splite_char:
                if component != "":
                    new_word += requests.get(setting["KOR2ENG_KOR"] + "/" + component).json()["eng"]
        except NotHangulException:
            new_word += char

    return new_word
