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

  if dataset_name in ["cic-ids-17-18-70", "cic-ids-17-18-70-S2", "cic-ids-17-18-70-S3"]: #e.g adult
    cfg.LABEL_NAME = "Label"
    cfg.ROOT_DIR = "cic_results_70"
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

  elif dataset_name in ["cic-ids-17-18-80"]: #e.g adult
    cfg.LABEL_NAME = "Label"
    cfg.ROOT_DIR = "cic_results_80"
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

  elif dataset_name in ["cover_quantile_shift", "cover_cluster_shift", "cover"]:
    cfg.LABEL_NAME = "Cover_Type"
    cfg.NUM_FEATURE = 54
    feature_names = [
        # Continuous features (10)
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",

        # Wilderness Area (4 binary)
        "Wilderness_Area1",
        "Wilderness_Area2",
        "Wilderness_Area3",
        "Wilderness_Area4",

        # Soil Type (40 binary)
        "Soil_Type1",
        "Soil_Type2",
        "Soil_Type3",
        "Soil_Type4",
        "Soil_Type5",
        "Soil_Type6",
        "Soil_Type7",
        "Soil_Type8",
        "Soil_Type9",
        "Soil_Type10",
        "Soil_Type11",
        "Soil_Type12",
        "Soil_Type13",
        "Soil_Type14",
        "Soil_Type15",
        "Soil_Type16",
        "Soil_Type17",
        "Soil_Type18",
        "Soil_Type19",
        "Soil_Type20",
        "Soil_Type21",
        "Soil_Type22",
        "Soil_Type23",
        "Soil_Type24",
        "Soil_Type25",
        "Soil_Type26",
        "Soil_Type27",
        "Soil_Type28",
        "Soil_Type29",
        "Soil_Type30",
        "Soil_Type31",
        "Soil_Type32",
        "Soil_Type33",
        "Soil_Type34",
        "Soil_Type35",
        "Soil_Type36",
        "Soil_Type37",
        "Soil_Type38",
        "Soil_Type39",
        "Soil_Type40",
    ]


    cfg.FEATURE_NAMES = feature_names
    cfg.DISCRETE_FEATURES  =  [
    # Wilderness Area (4)
    "Wilderness_Area1",
    "Wilderness_Area2",
    "Wilderness_Area3",
    "Wilderness_Area4",

    # Soil Type (40)
    "Soil_Type1",
    "Soil_Type2",
    "Soil_Type3",
    "Soil_Type4",
    "Soil_Type5",
    "Soil_Type6",
    "Soil_Type7",
    "Soil_Type8",
    "Soil_Type9",
    "Soil_Type10",
    "Soil_Type11",
    "Soil_Type12",
    "Soil_Type13",
    "Soil_Type14",
    "Soil_Type15",
    "Soil_Type16",
    "Soil_Type17",
    "Soil_Type18",
    "Soil_Type19",
    "Soil_Type20",
    "Soil_Type21",
    "Soil_Type22",
    "Soil_Type23",
    "Soil_Type24",
    "Soil_Type25",
    "Soil_Type26",
    "Soil_Type27",
    "Soil_Type28",
    "Soil_Type29",
    "Soil_Type30",
    "Soil_Type31",
    "Soil_Type32",
    "Soil_Type33",
    "Soil_Type34",
    "Soil_Type35",
    "Soil_Type36",
    "Soil_Type37",
    "Soil_Type38",
    "Soil_Type39",
    "Soil_Type40",
]

    cfg.STANDARDIZE = True
    cfg.NUM_CLASS = 7   




  elif dataset_name in ["poker", "poker_quantile_shift", "poker_cluster_shift"]:
    cfg.NUM_FEATURE = 10
    cfg.LABEL_NAME = "CLASS"
    feature_names = [
    "S1",  # Suit of card #1 (1–4)
    "C1",  # Rank of card #1 (1–13)
    "S2",  # Suit of card #2 (1–4)
    "C2",  # Rank of card #2 (1–13)
    "S3",  # Suit of card #3 (1–4)
    "C3",  # Rank of card #3 (1–13)
    "S4",  # Suit of card #4 (1–4)
    "C4",  # Rank of card #4 (1–13)
    "S5",  # Suit of card #5 (1–4)
    "C5",  # Rank of card #5 (1–13)
    ]
    cfg.FEATURE_NAMES = feature_names
    cfg.DISCRETE_FEATURES = [
    "S1",  # Suit of card #1 (1–4)
    "C1",  # Rank of card #1 (1–13)
    "S2",  # Suit of card #2 (1–4)
    "C2",  # Rank of card #2 (1–13)
    "S3",  # Suit of card #3 (1–4)
    "C3",  # Rank of card #3 (1–13)
    "S4",  # Suit of card #4 (1–4)
    "C4",  # Rank of card #4 (1–13)
    "S5",  # Suit of card #5 (1–4)
    "C5",  # Rank of card #5 (1–13)
    ]
    cfg.STANDARDIZE = False
    cfg.NUM_CLASS = 10

  elif dataset_name in ["shuttle", "shuttle_quantile_shift", "shuttle_cluster_shift"]:
    cfg.NUM_FEATURE = 7
    cfg.LABEL_NAME = "class"
    feature_names = [
    "Rad_Flow",
    "Fpv_Close",
    "Fpv_Open",
    "High",
    "Bypass",
    "Bpv_Close",
    "Bpv_Open",
    ]
    cfg.FEATURE_NAMES = feature_names
    cfg.DISCRETE_FEATURES = []
    cfg.STANDARDIZE = True
    cfg.NUM_CLASS = 7

  elif dataset_name in ["diabetes", "diabetes_quantile_shift", "diabetes_cluster_shift"]:
    cfg.NUM_FEATURE = 21
    cfg.LABEL_NAME = "Diabetes_binary"
    feature_names = [
    "HighBP",             # Binary: high blood pressure (0=no, 1=yes)
    "HighChol",           # Binary: high cholesterol (0=no, 1=yes)
    "CholCheck",          # Binary: cholesterol check in last year (0=no, 1=yes)
    "BMI",                # Integer: body mass index
    "Smoker",             # Binary: ever smoked 100 cigarettes (0=no, 1=yes)
    "Stroke",             # Binary: ever told you had a stroke (0=no, 1=yes)
    "HeartDiseaseorAttack", # Binary: heart disease or attack (0=no, 1=yes)
    "PhysActivity",       # Binary: physical activity in last 30 days (0=no, 1=yes)
    "Fruits",             # Binary: ate fruit 1+ times per day (0=no, 1=yes)
    "Veggies",            # Binary: ate vegetables 1+ times per day (0=no, 1=yes)
    "HvyAlcoholConsump",  # Binary: heavy alcohol consumption (0=no, 1=yes)
    "AnyHealthcare",      # Binary: has any health care coverage (0=no, 1=yes)
    "NoDocbcCost",        # Binary: did not see doctor due to cost (0=no, 1=yes)
    "GenHlth",            # Categorical: general health rating
    "MentHlth",           # Integer: days of poor mental health last 30 days
    "PhysHlth",           # Integer: days of poor physical health last 30 days
    "DiffWalk",           # Binary: difficulty walking (0=no, 1=yes)
    "Sex",                # Binary/Categorical: gender
    "Age",                # Integer: age category / bucketed ages
    "Education",          # Categorical: education level
    "Income"              # Categorical: income bracket
    ]
    cfg.FEATURE_NAMES = feature_names
    cfg.DISCRETE_FEATURES = [
          "HighBP",
          "HighChol",
          "CholCheck",
          "Smoker",
          "Stroke",
          "HeartDiseaseorAttack",
          "PhysActivity",
          "Fruits",
          "Veggies",
          "HvyAlcoholConsump",
          "AnyHealthcare",
          "NoDocbcCost",
          "DiffWalk",
          "Sex",
          "GenHlth",
          "Education",
          "Income",
          "Age"
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
