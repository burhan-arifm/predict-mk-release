from json import dumps as json_dumps
from logging import error as log_error

from azure.functions import HttpRequest, HttpResponse

from .exceptions import IncompleteCoursesError, ProgramNotFoundError
from .file_handler import handle_file
from .predict import recommend_matkul


def main(req: HttpRequest) -> HttpResponse:
    try:
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
    except IncompleteCoursesError as error:
        log_error(error)

        return HttpResponse(
            json_dumps({
                'error-type': 'incomplete-courses',
                'missing-items': error.missing_items
            }),
            status_code=454,
            headers={
                'Content-type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        )
    except ProgramNotFoundError as error:
        return HttpResponse(
            json_dumps({
                'error-type': 'program-unavailable'
            }),
            status_code=404,
            headers={
                'Content-type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        )
    except Exception as error:
        log_error(error)

        return HttpResponse(
            status_code=500,
            headers={
                'Content-type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        )
