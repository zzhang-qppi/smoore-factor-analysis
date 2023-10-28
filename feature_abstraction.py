import feature_utils as utils
import pandas as pd

count_data = pd.read_csv("label_counts.csv", index_col=0)

print(utils.XuQiuManZu(count_data))