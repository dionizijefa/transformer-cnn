import pandas as pd
import numpy as np

# Read the Excel file
df = pd.read_excel('data/US-EPA_OralRat_7405_moj2model_D-Fa2.xlsx')

# Separate the data based on the Dataset column
train_data = df[df['Dataset'] == 'Train']
val_data = df[df['Dataset'] == 'test1']
test_data = df[df['Dataset'] == 'test2']

# Select and rename columns
columns_mapping = {
    'SMILES': 'smiles',
    'LogLD50 {measured}': 'ld50'
}

train_data = train_data[columns_mapping.keys()].rename(columns=columns_mapping)
val_data = val_data[columns_mapping.keys()].rename(columns=columns_mapping)
test_data = test_data[columns_mapping.keys()].rename(columns=columns_mapping)

# Save to separate CSV files
train_data.to_csv('data/train_set.csv', index=False)
val_data.to_csv('data/validation_set.csv', index=False)
test_data.to_csv('data/test_set.csv', index=False)

# Print dataset sizes
print(f"Total samples: {len(df)}")
print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")