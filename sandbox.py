## explore data


import geopandas as gpd


training_shape_path = "raw_data/training_area/"

train_df = gpd.read_file(training_shape_path)

train_df.to_csv('train_label.csv', index = False)