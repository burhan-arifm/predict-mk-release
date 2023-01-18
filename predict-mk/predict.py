from operator import itemgetter
from os import remove
from os.path import dirname, join
from tempfile import gettempdir

from keras.models import load_model
from keras.utils import get_file
from numpy import average
from pandas import DataFrame, concat
from tensorflow import data, strings

from .exceptions import IncompleteCoursesError, ProgramNotFoundError

temp_dir = gettempdir()


def _setup_model(prodi):
    try:
        settings = {}
        # download the model and settings
        model_filepath = get_file(
            fname=f'model.zip',
            origin=f'https://predictmk2.blob.core.windows.net/models/{prodi}v2.zip',
            cache_dir=temp_dir,
            cache_subdir='model',
            extract=True
        )
        model_path = dirname(model_filepath)
        remove(model_filepath)
        # load the model
        model = load_model(model_path)
        # load the settings
        settings_path = join(model_path, 'assets', 'settings')
        with open(settings_path, encoding='utf8') as f:
            for line in f:
                (key, value) = line.replace('\n', '').split('=')
                settings[key] = value.split(';')
                if len(settings[key]) <= 1:
                    settings[key] = settings[key][0]
                    if settings[key].isnumeric():
                        settings[key] = int(settings[key])
                else:
                    for i, value in enumerate(settings[key]):
                        settings[key][i] = str(value)

        return model, settings
    except Exception as error:
        if str(error).find('404') != -1:
            raise ProgramNotFoundError()


def _create_sequences(values, window_size, step_size, targets):
    sequences = []
    for target in targets:
        start_index = 0
        end_index = 0
        while end_index < len(values):
            end_index = start_index + window_size - 1
            seq = values[start_index:end_index]
            if len(seq) < window_size - 1:
                seq = values[-(window_size - 1):]
            seq.append(target)
            sequences.append(seq)
            start_index += step_size

    return sequences


def _change_last_value(sequences):
    for sequence in sequences:
        sequence.pop()
        sequence.append(0)

    return sequences


def _get_dataset_from_csv(csv_file_path, column_names=[], shuffle=False, batch_size=128):
    def process(features):
        kode_mk_string = features['seq_matkul']
        seq_matkul = strings.split(kode_mk_string, ',').to_tensor()

        features['target_matkul'] = seq_matkul[:, -1]
        features['seq_matkul'] = seq_matkul[:, :-1]

        nilai_string = features['seq_nilai']
        seq_nilai = strings.to_number(
            strings.split(nilai_string, ',')).to_tensor()

        target = seq_nilai[:, -1]
        features['seq_nilai'] = seq_nilai[:, :-1]

        return features, target

    dataset = data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=column_names,
        num_epochs=1,
        header=False,
        field_delim='|',
        shuffle=shuffle
    ).map(process)

    return dataset


def _check_dataset_completeness(dataset: DataFrame, compare_to: list):
    missing_courses = []
    column_names = []
    all_courses = DataFrame(data=compare_to, columns=['kode_mk'])

    # Courses not completed because of score not given but took the course
    for (_, matkul, nilai) in dataset.itertuples(index=False, name=None):
        if isinstance(nilai, str):
            missing_courses.append(matkul)

    # Courses not completed because of not take the class
    column_names = dataset.columns.to_list()
    column_names.append('_merge')
    outer = all_courses.merge(
        dataset, how='outer', left_on='kode_mk', right_on='matkul', indicator=True)
    incomplete = outer[(outer._merge == 'left_only')].drop(
        columns=column_names)
    missing_courses = incomplete['kode_mk'].to_list()

    if missing_courses:
        raise IncompleteCoursesError(missing_courses)


def recommend_matkul(data):
    dataset_filepath = join(temp_dir, 'temp.csv')
    kode_prodi = data[0][0][3:6]

    # Setup the models and parameters
    model, settings = _setup_model(kode_prodi)

    # Create DataFrame
    nilai = DataFrame(data, columns=['nim', 'matkul', 'nilai'])

    # Check completeness
    _check_dataset_completeness(nilai.copy(
    ), compare_to=settings['MK_TRANSLATE_SOURCE'] if settings['KODE_MK_TRANSLATED'] else settings['MK_TRANSLATE_TARGET'])

    # Preprocessing dataset
    nilai = nilai[~nilai['matkul'].isin(settings['REMOVED_MK'])]
    nilai['nim'] = nilai['nim'].apply(lambda x: f'mhs_{x}')
    if settings['KODE_MK_TRANSLATED']:
        nilai['matkul'] = nilai['matkul'].replace(
            to_replace=settings['MK_TRANSLATE_SOURCE'],
            value=settings['MK_TRANSLATE_TARGET']
        )
    nilai_group = nilai.groupby('nim')
    nilai_data = DataFrame(
        data={
            'nim': list(nilai_group.groups.keys()),
            'matkul': list(nilai_group.matkul.apply(list)),
            'nilai': list(nilai_group.nilai.apply(list))
        }
    )

    # Creating sequences
    nilai_data.matkul = nilai_data.matkul.apply(
        lambda x: _create_sequences(
            x,
            settings['SEQUENCE'],
            settings['STEP'],
            settings['KODE_MK_TARGET']
        )
    )
    nilai_data.nilai = nilai_data.nilai.apply(
        lambda x: _change_last_value(
            _create_sequences(
                x,
                settings['SEQUENCE'],
                settings['STEP'],
                settings['KODE_MK_TARGET']
            )
        )
    )
    nilai_data_matkul = nilai_data[['nim', 'matkul']
                                   ].explode('matkul', ignore_index=True)
    nilai_data_nilai = nilai_data[['nilai']].explode(
        'nilai', ignore_index=True)
    nilai_data_transformed = concat(
        [nilai_data_matkul, nilai_data_nilai], axis=1)
    nilai_data_transformed.matkul = nilai_data_transformed.matkul.apply(
        lambda x: ','.join(x))
    nilai_data_transformed.nilai = nilai_data_transformed.nilai.apply(
        lambda x: ','.join([str(v) for v in x]))
    nilai_data_transformed.rename(
        columns={'matkul': 'seq_matkul', 'nilai': 'seq_nilai'},
        inplace=True
    )
    nilai_data_transformed.to_csv(
        dataset_filepath, index=False, sep='|', header=False)

    # # Creating prediction dataset
    CSV_HEADER = list(nilai_data_transformed.columns)
    csv_input = _get_dataset_from_csv(
        dataset_filepath, column_names=CSV_HEADER, batch_size=256)

    # Get prediction
    prediction = model.predict(csv_input)

    # Post processing prediction
    kode_mk_target = settings['KODE_MK_TARGET_REAL'] if settings['KODE_MK_TRANSLATED'] else settings['KODE_MK_TARGET']
    prediction = prediction.reshape(5, int(prediction.shape[0]/5))
    prediction = average(prediction, axis=1)
    prediction_combined = list(
        map(list, zip(kode_mk_target, settings['MK_TARGET'], prediction)))
    sorted_prediction = sorted(
        prediction_combined, key=itemgetter(2), reverse=True)
    result = [' - '.join(tup[:-1]) for tup in sorted_prediction[:3]]

    return result
