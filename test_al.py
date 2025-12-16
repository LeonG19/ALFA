from pool_filters_cumulative import subpool_anchoral, subpool_randsub, subpool_seals
import pandas as pd
from sklearn.model_selection import train_test_split
df_2017 = pd.read_csv("raw_data/CIC_2017_day_aligned.csv")
df_2018 = pd.read_csv("raw_data/CIC_2018_day_aligned.csv")
print(df_2017.shape)
X_labeled_2017, X_unlabeled_2017, y_labeled_2017, y_unlabeled_2017 = train_test_split(df_2017.drop(columns =["Label"]), df_2017["Label"], test_size = .5, random_state=42)
X_labeled_2018, X_unlabeled_2018, y_unlabeled_2018, y_labeled_2018  = train_test_split(df_2018.drop(columns = ["Label"]), df_2018["Label"], test_size = .5, random_state=42)

#run_demo_round(X_labeled_2017, y_labeled_2017, X_unlabeled_2018, y_labeled_2018, M = 10000)

X_sub, y_sub = subpool_anchoral(X_labeled_2017, y_labeled_2017, X_unlabeled_2018, y_unlabeled_2018)
X_sub, y_sub = subpool_randsub(X_labeled_2017, y_labeled_2017, X_unlabeled_2018, y_unlabeled_2018)
X_sub, y_sub = subpool_seals(X_labeled_2017, y_labeled_2017, X_unlabeled_2018, y_unlabeled_2018)
