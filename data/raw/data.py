import pandas as pd

# Đọc file CSV 1 và CSV 2
df1 = pd.read_csv(r"data_1.csv")
df2 = pd.read_csv(r"data_2.csv")

key_cols = ['Water Control Zone', 'Station', 'Dates', 'Sample No']

# Ghép hai file theo khóa
merged_df = pd.merge(df1, df2, on=key_cols, how='outer', suffixes=('_sediment', '_water'))

# Lưu file kết quả
merged_df.to_csv(r"C:\Users\Admin\Desktop\Data_science\data\raw\merged_output.csv", index=False)

print("Đã ghép file thành công, lưu tại merged_output.csv")
