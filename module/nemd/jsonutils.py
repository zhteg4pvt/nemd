import json


def load(file):
    """
    Return the loaded data.

    :return dict: the data in the job json.
    """
    try:
        with open(file) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        return {}
