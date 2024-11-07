import pandas as pd

def check_purchase_and_category(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Check if the necessary columns exist in the DataFrame
    if 'event_type' in df.columns and 'category_code' in df.columns:
        # Check for the specific condition
        condition_exists = df[(df['event_type'] == 'purchase') & (df['category_code'] == 'apparel.shoes')].any().any()
        if condition_exists:
            print("There is at least one entry with event_type = 'purchase' and category_code = 'apparel.shoes'.")
        else:
            print("There are no entries with event_type = 'purchase' and category_code = 'apparel.shoes'.")
    else:
        print("The necessary columns ('event_type' and 'category_code') do not exist in the CSV file.")

if __name__ == "__main__":
    # Specify the path to your CSV file
    csv_file_path = 'data/2019-Nov.csv'
    
    # Check for the specific condition
    check_purchase_and_category(csv_file_path)
