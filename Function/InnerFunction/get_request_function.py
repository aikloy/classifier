import json


# Flask request body 또는 param 중 있는 데이터를 가져옴
def get_request(request):
    if len(request.data) > 0:
        data = json.loads(str(request.data, "utf-8"))
        if data:
            return data

    if len(request.values) > 0:
        data = {key: request.args.get(key) for key in request.values}
        if data:
            return data

    return None

