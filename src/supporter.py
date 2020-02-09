import os


def string_edit(string, flag):      # function accepts a string and a flag value 0 or 1, returns the edited path
    path_to_legacy_core = os.getcwd()
    path_to_legacy = os.path.split(path_to_legacy_core)[0]
    path = os.path.join(path_to_legacy, string)
    path = path.replace("\\", "/")
    if flag == 1:
        path_with_quotes = f'"{path}"'
        return path_with_quotes
    elif flag == 0:
        return path


def concat(string1, string2):
    string = str(string1) + str(string2)
    return string


def folder_paths(train_csv, test_csv, submission_csv):
    path_to_src = os.getcwd()
    path_to_input = string_edit('input/', 0)
    path_to_models = string_edit('models/', 0)
    path_to_train_csv = concat(path_to_input, train_csv)
    path_to_test_csv = concat(path_to_input, test_csv)
    path_to_submission_csv = concat(path_to_input, submission_csv)
    return {"path_to_src": path_to_src,
            "path_to_input": path_to_input,
            "path_to_train_csv": path_to_train_csv,
            "path_to_test_csv": path_to_test_csv,
            "path_to_submission_csv": path_to_submission_csv,
            "path_to_models": path_to_models}