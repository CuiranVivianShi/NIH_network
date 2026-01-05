import pandas as pd

# Load your data
df = pd.read_csv('PI_Info_Updated_Reporter12_largest_component.csv')

# Assuming 'Title' is the column you want to split
# Expand the split results into separate columns
title_df = df['Title'].str.split(',', expand=True)

# You can name these columns based on your preference or keep them generic
column_names = [f'Column_{i}' for i in range(title_df.shape[1])]
title_df.columns = column_names

# Join the new columns back to the original dataframe
df = df.join(title_df)

# Optionally, drop the original 'Title' column if it's no longer needed
# df.drop('Title', axis=1, inplace=True)

# Save the result to a new CSV file
df.to_csv('PI_Info_Updated_Reporter12_largest_component.csv', index=False)
