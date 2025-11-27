import os
from hyper_parameter_optimizer.optimizer import HyperParameterOptimizer
import pandas as pd


def link_fieldnames_data(_config):
    root_path = _config['root_path']
    data_path = _config['data_path']
    data = pd.read_csv(os.path.join(root_path, data_path))
    feature_length = len(data.columns) - 1
    _config['enc_in'] = feature_length
    _config['dec_in'] = feature_length
    _config['c_out'] = feature_length
    return _config


# noinspection DuplicatedCode
def get_search_space():
    default_config = {
        'task_name': {'_type': 'single', '_value': 'long_term_forecast'},
        'is_training': {'_type': 'single', '_value': 0},
        # 'test_checkpoint_path': {'_type': 'single',
        #                          '_value': 'save/maml_wind_5_way_5_shot_2025_11_24_13_16_57/min-vl.pth'},
        'test_checkpoint_path': {'_type': 'single',
                                 '_value': 'save/maml_Penmanshiel_5_way_5_shot_2025_11_27_17_16_57/min-vl.pth'},
        'des': {'_type': 'single', '_value': 'Exp'},
        'use_gpu': {'_type': 'single', '_value': True},
        'embed': {'_type': 'single', '_value': 'timeF'},
        'freq': {'_type': 'single', '_value': 't'},
        'batch_size': {'_type': 'single', '_value': 32},
        'pin_memory': {'_type': 'single', '_value': False},
    }

    dataset_config = {
        'data': {'_type': 'single', '_value': 'custom'},
        'features': {'_type': 'single', '_value': 'MS'},
        # wind zone
        # 'root_path': {'_type': 'single', '_value': './.materials/wind/'},
        # 'data_path': {'_type': 'choice', '_value': [f"Zone{i}/Zone{i}.csv" for i in range(1, 11)]},
        # 'target': {'_type': 'single', '_value': 'wind'},
        # penmanshiel turbine
        'root_path': {'_type': 'single', '_value': './.materials/Penmanshiel_SCADA_2022_WT01-15/'},
        'data_path': {'_type': 'choice', '_value': [f"filtered_Turbine_Data_Penmanshiel_{i:02d}_2022-01-01_-_2023-01-01.csv"
                                                    for i in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]},
        'target': {'_type': 'single', '_value': 'Power (kW)'},
        'enc_in': {'_type': 'single', '_value': 5},
        'dec_in': {'_type': 'single', '_value': 5},
        'c_out': {'_type': 'single', '_value': 5},
    }

    learning_config = {
        'learning_rate': {'_type': 'single', '_value': 0.0001},
        'train_epochs': {'_type': 'single', '_value': 100},
        'patience': {'_type': 'single', '_value': 15},
    }

    period_config = {
        'seq_len': {'_type': 'single', '_value': 96},
        'label_len': {'_type': 'single', '_value': 96},
        'pred_len': {'_type': 'single', '_value': 16},
        'moving_avg': {'_type': 'single', '_value': 25},
    }

    dlinear_config = {
        'd_model': {'_type': 'single', '_value': 128},
        # 'individual': {'_type': 'single', '_value': True},
    }

    model_configs = {
        'DLinear': dlinear_config,
    }

    return [default_config, dataset_config, learning_config, period_config], model_configs


h = HyperParameterOptimizer(script_mode=False, models=['DLinear'],
                            get_search_space=get_search_space, link_fieldnames_data=link_fieldnames_data)
h.config_optimizer_settings(root_path='.', data_csv_file='Penmanshiel2022_6112.csv',
                            scan_all_csv=False, try_model=False, force_exp=False)
