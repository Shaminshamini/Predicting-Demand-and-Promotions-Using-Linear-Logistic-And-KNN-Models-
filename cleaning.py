import pandas as pd

# 1️ Load raw dataset
df = pd.read_csv("sales_data.csv")

# 2️ Remove duplicates
df = df.drop_duplicates()

# 3️ Correct any typos in Promotion column
df['Promotion'] = df['Promotion'].replace('Pr0motion', 'Promotion')

# 4️ Fix negative Discount and Price values
df['Discount'] = df['Discount'].clip(lower=0, upper=100)  # Discount must be between 0-100%
df['Price'] = df['Price'].clip(lower=0)  # Price cannot be negative

# 5 Convert 'Date' column to datetime if it exists
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Month'] = df['Date'].dt.month

# 6️ Encode categorical columns (Promotion and Epidemic)
df['Promotion'] = df['Promotion'].map({'Promotion': 1, 'No': 0})
df['Epidemic'] = df['Epidemic'].map({'Yes': 1, 'No': 0})

# 7️ Handle any remaining missing values (just in case)
df = df.fillna(0)

# 8️ Save cleaned data
df.to_csv("cleaned_sales_data.csv", index=False)

print("✅ Data cleaning completed successfully!")
