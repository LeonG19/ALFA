from yacs.config import CfgNode as CN
import pandas as pd
import numpy as np
def get_dataset_config(dataset_name):
  cfg = CN()
  cfg.NAME = dataset_name
  cfg.NUM_WORKERS = 2
  cfg.DATA_DIR = 'data/' + dataset_name
  cfg.TRAIN_FILE = 'train.npz'
  cfg.VAL_FILE = 'val.npz'
  cfg.TEST_FILE = 'test.npz'
  cfg.DUPLICATES = 'remove-cross'
  cfg.CACHE = True
  cfg.TRANSFORM = None
  cfg.IS_PYTORCH = False  # whether it will be used to train a PyTorch model or not (i.e., skelarn/XGBoost)
  cfg.VAL_BATCH_SIZE = 64  # for neural nets

  if dataset_name in ['cic-ids-17-18', 'cic-ids-18-17', "cic-ids-17-18-70"]: #e.g adult
    cfg.LABEL_NAME = "Label"

    df= pd.read_csv("raw_data/CIC_2017_day_aligned.csv")
    feature_names = (
       df
       .drop(columns = "Label")                     
       .columns
       .tolist()
    )
    cfg.FEATURE_NAMES = feature_names
    cfg.DISCRETE_FEATURES = [    "Dst Port",
    "Total Fwd Packet",
    "Total Bwd packets",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd RST Flags",
    "Bwd RST Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWR Flag Count",
    "ECE Flag Count",
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
    "FWD Init Win Bytes",
    "Bwd Init Win Bytes",
    "ICMP Code",
    "ICMP Type",
    "Label"
    ]
    cfg.NUM_FEATURE = 82
    cfg.STANDARDIZE =  True                                                                      
    cfg.NUM_CLASS = 9


  elif dataset_name == "adult":
    cfg.NUM_FEATURE = 14
    
    df= pd.read_csv("raw_data/adult.csv")
    feature_names = (
       df
       .drop(columns = "income")                     
       .columns
       .tolist()
    )

    cfg.FEATURE_NAMES = feature_names
    cfg.DISCRETE_FEATURES  = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "native-country",
    "gender",
    "Label",
    ]
    cfg.STANDARDIZE = True
    cfg.NUM_CLASS = 2


  else:
    raise ValueError('unknown dataset {}'.format(dataset_name))

  return cfg


def get_model_config(model_name):
  cfg = CN()
  cfg.MODEL_NAME = model_name
  if model_name == 'xgboost':
    cfg.XGBOOST = CN()
    cfg.XGBOOST.USE_GPU = False
  elif model_name == 'mlp':
    cfg.MLP = CN()
    cfg.MLP.USE_GPU = True
  elif model_name == 'scc':
    cfg.SCC = CN()
    cfg.SCC.USE_GPU = True
    
  return cfg


def get_experiment_cfg():
  cfg = CN()
  cfg.SEED = 1
  cfg.OUT_DIR = 'output'
  cfg.VALIDATION_MODE = True
  cfg.NUM_HPO_RUNS = 1  # Number of runs for each hyperparameter (for different random initialization)
  cfg.NUM_TEST_RUNS = 5   # run with 5 random seeds for final hyperparameters
  cfg.TASK = 'classification'
  cfg.PARAMS = None  # file containing model hyperparams, can be passed to instantiate parameter from a file

  cfg.ACTIVE_LEARNING_SAMPLES = 50
  return cfg


def get_default_config():
  cfg = CN()
  cfg.EXPERIMENT = get_experiment_cfg()

  cfg.EXPERIMENT.MODEL_NAME = 'xgboost'
  cfg.DATASET = get_dataset_config('androzoo-drebin')
  cfg.MODEL = get_model_config(cfg.EXPERIMENT.MODEL_NAME)
  return cfg


def get_config(dataset, model):
  cfg = CN()
  cfg.EXPERIMENT = get_experiment_cfg()

  cfg.EXPERIMENT.MODEL_NAME = model
  cfg.DATASET = get_dataset_config(dataset)
  return cfg


if __name__ == '__main__':
  cf = get_dataset_config('androzoo-apigraph')
  print(cf)
