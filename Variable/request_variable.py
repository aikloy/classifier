REQUEST_BASE = "http://{ip}:{port}{url}"
REQUEST_BASE_WITH_URL = "https://{base_url}{url}"

RESPONSE_OK = 200
RESPONSE_OK_WITH_NO_CONTENT = 204
RESPONSE_PARTIAL_OK = 206
RESPONSE_INVALID_INPUT = 400
RESPONSE_INTERNAL_ERROR = 500

MODEL_NOT_EXIST = {
    "error": "model not exist"
}
TRAIN_DATA_NOT_EXIST = {
    "error": "index not exist"
}

TRAIN = {
    "error": "Invalid input",
    "valid_input": {
        "scope": "string / (scope name)"
    }
}
TEST = {
    "error": "Invalid input",
    "valid_input": {}
}
FORWARD = {
    "error": "Invalid input",
    "valid_input": {
        "scope": "string / (scope name)",
        "msg": "string / (massage)"
    }
}
