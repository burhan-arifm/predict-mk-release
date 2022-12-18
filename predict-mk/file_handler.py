from base64 import b64encode
from os import remove
from tempfile import gettempdir
from uuid import uuid4

from aspose.words import Document
from aspose.words.loading import PdfLoadOptions
from imagekitio import ImageKit

imagekit = ImageKit(
    private_key='private_wMXOW196RlUYGEqwclebmAJ20mM=',
    public_key='public_JIr32fHfoZXd0rR/WcY/Na2dlJg=',
    url_endpoint='https://ik.imagekit.io/burhanarifm/mk-transcript/'
)
temp_dir = gettempdir()
pdf_load_options = PdfLoadOptions()
pdf_load_options.skip_pdf_images = True


def _parse_markdown(file_path):
    lines = []
    dataset = []
    with open(file_path, encoding='utf8') as file:
        lines = [str(line) for line in file]

    nim = lines[18].split(' ')[2]
    data = [line.split('|')[2:-2] for line in lines[28:-6]]

    for row in data:
        matkul = row[0]
        nilai = 0.00

        if row[-1] != '':
            nilai = float(row[-1])

        dataset.append([nim, matkul, nilai])

    return dataset


def _pdf_to_md(file_url):
    document = Document(file_url, load_options=pdf_load_options)
    file_output_path = f'{temp_dir}/{uuid4()}.md'

    document.save(file_output_path)

    return file_output_path


def _upload_file(file_name, content):
    content = b64encode(content)

    result = imagekit.upload_file(
        file=content,
        file_name=file_name
    )

    return result.file_id, result.url


def handle_file(file_name, content):
    file_id, file_url = _upload_file(file_name, content)
    path = _pdf_to_md(file_url)
    dataset = _parse_markdown(path)

    # cleanups
    imagekit.delete_file(file_id)
    remove(path)

    return dataset
