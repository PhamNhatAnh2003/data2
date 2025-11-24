import pandas as pd

# Đọc file CSV 1 và CSV 2
df1 = pd.read_csv("path/to/file1.csv")
df2 = pd.read_csv("path/to/file2.csv")

# Danh sách các cột dùng làm khóa để ghép
key_cols = ['Water Control Zone', 'Station', 'Dates', 'Sample No']

# Ghép 2 dataframe theo khóa
merged_df = pd.merge(df1, df2, on=key_cols, how='outer', suffixes=('_water', '_sediment'))

# Lưu file kết quả
merged_df.to_csv("merged_output.csv", index=False)
