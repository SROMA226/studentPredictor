# Considerer train.csv et test.csv
# Extraire les caracteriques qui nous intereste
# Enregistrer

import pandas as pd

from config import Config

Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)

df_train = pd.read_csv(str(Config.DATASET_PATH / "train.csv"))
df_test = pd.read_csv(str(Config.DATASET_PATH / "test.csv"))

features = ['is_paid', 'price', 'num_reviews', 'num_lectures', 'level', 'content_duration']
levels_map = {
    'All Levels': 1, 
    'Beginner Level': 2,
    'Intermediate Level': 3, 
    'Expert Level': 4
}
def extract_features(df):
    # Transforme True en 1 et False en 0
    df['is_paid'] = df['is_paid'].astype(int)

    # Transformer les valeurs de level en leur equivalent dans `levels_map`
    def process_level(row):
        return levels_map[row['level']]
        
    df['level'] = df.apply(lambda row: process_level(row), axis=1)

    return df[features]

train_features = extract_features(df_train)
test_features = extract_features(df_test)

# Enregistrement des features pour train et test
train_features.to_csv(str(Config.FEATURES_PATH / "train_features.csv"), index=None)
test_features.to_csv(str(Config.FEATURES_PATH / "test_features.csv"), index=None)

# Enregistrement des labels pour train et test
df_train.num_subscribers.to_csv(str(Config.FEATURES_PATH / "train_labels.csv"), index=None)
df_test.num_subscribers.to_csv(str(Config.FEATURES_PATH / "test_labels.csv"), index=None)
