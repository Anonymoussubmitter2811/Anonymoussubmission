DATASETS_LOCATION = {
    "musk1": 'CONFIGS/Musk/originalData/clean1.data',
    "musk2": "CONFIGS/Musk/originalData/clean2.data",
    "fox": "CONFIGS/Fox/data_100x100.svm",
    "tiger": "CONFIGS/Tiger/data_100x100.svm",
    "elephant": "CONFIGS/Elephant/data_100x100.svm"
}

DATASETS_PARAMS = {

    'musk1':
        {
            "lr": 0.05,
            "num_epochs": 10000,
            'validation_rate': 0.2,
            'wd': 0.01,
            'drop_out' : 0
        },
    'musk2':
        {
            "lr": 0.01,
            "num_epochs": 10000,
            'validation_rate': 0.2,
            'wd': 0.01,
            'drop_out' : 0
        },
    'fox':
        {
            "lr": 0.01,
            "num_epochs": 10000,
            'validation_rate': 0.2,
            'wd': 0.045,
            'drop_out' : 0.2
        },
    'tiger':
        {
            "lr": 0.008,
            "num_epochs": 10000,
            'validation_rate': 0.2,
            'wd': 0.08,
            'drop_out' : 0.3
        },
    'elephant':
        {
            "lr": 0.001,
            "num_epochs": 10000,
            'validation_rate': 0.2,
            'wd': 0.001,
            'drop_out':0.2
        }
}


def get_params(argv: list):
    """
    sys.argv[1] = dataset name
    """
    if len(argv) == 0:
        raise Exception("No arguments given. Expects at least 1")
    params = {"dataset_name": argv[0]}
    return params
