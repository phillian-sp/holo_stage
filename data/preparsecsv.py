import pandas as pd
from tqdm import tqdm

# Define the chunk size
chunk_size = 100000

# List of category codes to filter
category_codes = [
    'computers.desktop', 
    'appliances.kitchen.refrigerators', 
    'furniture.bedroom.bed', 
    'electronics.smartphone', 
    'apparel.shoes'
]

# Get the total number of rows in the CSV file for progress bar
total_rows = sum(1 for _ in open('2019-Nov.csv')) - 1  # Subtract 1 for the header

# Process the CSV file in chunks with a progress bar
for chunk in tqdm(pd.read_csv('2019-Nov.csv', chunksize=chunk_size), total=total_rows // chunk_size + 1):
    # Filter the rows where event_type is 'purchase' and category_code is in the list
    filtered_chunk = chunk[(chunk['event_type'] == 'purchase') & (chunk['category_code'].isin(category_codes))]
    
    # Append the filtered rows to the respective category CSV file
    for category in category_codes:
        category_chunk = filtered_chunk[filtered_chunk['category_code'] == category]
        if not category_chunk.empty:
            category_chunk.to_csv(f'{category.replace(".", "_")}.csv', mode='a', header=False, index=False)
