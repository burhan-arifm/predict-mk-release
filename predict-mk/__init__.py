from json import dumps as json_dumps

from azure.functions import HttpRequest, HttpResponse

from .file_handler import handle_file
from .predict import recommend_matkul


def main(req: HttpRequest) -> HttpResponse:
    dataset = []
    for input_file in req.files.values():
        file_name = input_file.filename
        content = input_file.stream.read()
        data = handle_file(file_name, content)

        for item in data:
            dataset.append(item)

    recommendations = recommend_matkul(data=dataset)
    response = json_dumps({
        'result': recommendations
    })
    response_headers = {
        'Content-type': 'application/json',
        'Access-Control-Allow-Origin': '*'
    }

    return HttpResponse(
        response,
        status_code=200,
        headers=response_headers
    )
