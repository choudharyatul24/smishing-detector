import pandas as pd
import re
from datetime import datetime

# Dataset paths
paths = [
    r"C:\Users\choud\OneDrive\Documents\Datasets\SMSSpamCollection",
    r"C:\Users\choud\OneDrive\Documents\Datasets\ExAIS_SMS_SPAM_DATA.csv",
    r"C:\Users\choud\OneDrive\Documents\Datasets\Dataset_5971\Dataset_5971.csv"
]

dfs = []

# Load each dataset
for path in paths:
    if path.endswith("SMSSpamCollection"):
        df = pd.read_csv(path, sep="\t", header=None)
    else:
        df = pd.read_csv(path)
    
    # Standardize first two columns to 'message' and 'label'
    cols = df.columns.tolist()
    if len(cols) >= 2:
        df.rename(columns={cols[0]: 'message', cols[1]: 'label'}, inplace=True)
    
    dfs.append(df)

# Combine all datasets
combined_df = pd.concat(dfs, ignore_index=True)

# Keep only 'message' and 'label' columns
combined_df = combined_df[['message', 'label']]

# Drop rows with missing values
combined_df = combined_df.dropna(subset=['message', 'label']).reset_index(drop=True)

# Drop duplicate messages
combined_df = combined_df.drop_duplicates(subset=['message']).reset_index(drop=True)

# Map labels to numeric: spam=1, ham/others=0
combined_df['label_num'] = combined_df['label'].map(lambda x: 1 if 'spam' in str(x).lower() else 0)

# Clean text: lowercase, remove URLs, punctuation, extra spaces
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)      # remove URLs
    text = re.sub(r'[^a-z0-9\s]', '', text) # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()# remove extra spaces
    return text

combined_df['message'] = combined_df['message'].apply(clean_text)

# Save combined dataset with timestamped filename to avoid permission issues
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = fr"C:\Users\choud\OneDrive\Documents\flask\combined_dataset_{timestamp}.csv"
combined_df.to_csv(output_path, index=False)

print(f"✅ Combined dataset saved to: {output_path}")
print("\nPreview of dataset:\n", combined_df.head())
print("\nLabel distribution:\n", combined_df['label_num'].value_counts())
