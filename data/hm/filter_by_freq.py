import pandas as pd

# Define the value of k
k = 300  # you can set k to any desired value

# Read the CSV files
transactions = pd.read_csv('part_transactions.csv')
customers = pd.read_csv('customers.csv')
articles = pd.read_csv('articles.csv')

# Calculate the frequencies of article_id and customer_id
article_freq = transactions['article_id'].value_counts().head(k)
customer_freq = transactions['customer_id'].value_counts().head(k)

# Print the exact frequencies of the top k article_id and customer_id
print("Top k article_id frequencies:")
print(article_freq)
print(f"\nSum of top k article_id frequencies: {article_freq.sum()}")

print("\nTop k customer_id frequencies:")
print(customer_freq)
print(f"\nSum of top k customer_id frequencies: {customer_freq.sum()}")

# Get the top k frequent article_id and customer_id
top_k_articles = article_freq.index
top_k_customers = customer_freq.index

# Filter the rows in customers.csv and articles.csv
filtered_customers = customers[customers['customer_id'].isin(top_k_customers)]
filtered_articles = articles[articles['article_id'].isin(top_k_articles)]

# Save the filtered data to new CSV files
filtered_customers.to_csv('top_k_frequent_customers.csv', index=False)
filtered_articles.to_csv('top_k_frequent_articles.csv', index=False)

print("Files saved as 'top_k_frequent_customers.csv' and 'top_k_frequent_articles.csv'")
