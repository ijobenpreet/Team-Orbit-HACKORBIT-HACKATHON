import pandas as pd

# Load Kaggle dataset
try:
    df_kaggle = pd.read_csv("train.csv")
except FileNotFoundError:
    print("Error: train.csv not found. Download from Kaggle and place in 'C:\\Users\\hp\\OneDrive\\sukhman files\\projects\\team_orbit\\'.")
    exit(1)

# Handle missing/empty values
df_kaggle = df_kaggle.dropna(subset=["comment_text"])  # Remove rows with NaN
df_kaggle = df_kaggle[df_kaggle["comment_text"].str.strip() != ""]  # Remove empty or whitespace-only strings
df_kaggle = df_kaggle[~df_kaggle["comment_text"].isin(["", " "])]  # Explicitly drop "" or " "

# Map Kaggle labels to your categories
df_kaggle["hate_speech"] = df_kaggle["identity_hate"] | df_kaggle["toxic"]
df_kaggle["cyberbullying"] = df_kaggle["insult"] | df_kaggle["toxic"]
df_kaggle["incitement_violence"] = df_kaggle["threat"]
df_kaggle["threat_safety"] = df_kaggle["threat"] | df_kaggle["severe_toxic"]

# Select relevant columns (excluding fake_account for training)
df_kaggle = df_kaggle[["comment_text", "hate_speech", "cyberbullying", "incitement_violence", "threat_safety"]]
df_kaggle.rename(columns={"comment_text": "text"}, inplace=True)

# Subset for faster training
df_kaggle = df_kaggle.sample(n=10000, random_state=42)  # Adjust to 10,000 if needed

# Oversample rare labels
positive_samples = df_kaggle[df_kaggle["threat_safety"] == 1]
df_kaggle = pd.concat([df_kaggle, positive_samples.sample(n=500, replace=True)], ignore_index=True)

# Save as training_data.csv
df_kaggle.to_csv("training_data.csv", index=False)
print("Processed Kaggle dataset saved to training_data.csv")