import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
from concurrent.futures import ThreadPoolExecutor

def get_pi_department(pi_name):
    base_url = "https://pubmed.ncbi.nlm.nih.gov"
    # Split the name and use only the last name and the first name for the query
    parts = pi_name.split(',')
    last_name = parts[0]
    first_name = parts[1].strip().split()[0] if len(parts) > 1 and len(parts[1].strip().split()) > 0 else ''
    query_name = f"{last_name}, {first_name}"
    query = urllib.parse.quote_plus(query_name)
    url = f"{base_url}?term={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    try:
        latest_article_link = None
        latest_year = 0
        articles = soup.find_all('div', class_='docsum-wrap')

        for article in articles:
            citation = article.find('span', class_='docsum-journal-citation full-journal-citation')
            year = max([int(s) for s in citation.text.split() if s.isdigit()], default=0)
            if year > latest_year:
                link_tag = article.find('a', class_='docsum-title')
                if link_tag and link_tag.has_attr('href'):
                    latest_article_link = f"https://pubmed.ncbi.nlm.nih.gov{link_tag['href']}"
                    latest_year = year

        if latest_article_link:
            article_response = requests.get(latest_article_link)
            article_soup = BeautifulSoup(article_response.text, 'html.parser')
            authors = article_soup.find_all('span', class_='authors-list-item')
            for author in authors:
                author_name_element = author.find('a', class_='full-name')
                last_name = author_name_element.text.split()[-1]
                if last_name.lower() == query_name.split(',')[0].lower():
                    affiliation_link = author.find('a', class_='affiliation-link')
                    if affiliation_link:
                        return affiliation_link.get('title')
            return "No valid affiliation found"
    except Exception as e:
        return f"Error: {str(e)}"

def fetch_all_departments(pis):
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for i in range(0, len(pis), 10):
            partial_results = list(executor.map(get_pi_department, pis[i:i+10]))
            results.extend(partial_results)
            print(f"Processed {i + 10} PIs, taking a pause...")
            time.sleep(5)  # Pause for 5 seconds every 10 requests
    return results

# Load existing data
pi_info_df = pd.read_csv('PI_Info_Updated_Reporter6_largest_component.csv')

# Find rows with blank 'Title'
blank_indices = pi_info_df[pi_info_df['Title'].isna() | (pi_info_df['Title'] == "Error: 'NoneType' object has no attribute 'text'") | (pi_info_df['Title'] == "No valid affiliation found")].index.tolist()

# List of PIs to rescrape
pis_to_rescrape = pi_info_df.loc[blank_indices, 'PI Name'].tolist()

# Rescrape data for these PIs
new_titles = fetch_all_departments(pis_to_rescrape)

# Update the DataFrame
pi_info_df.loc[blank_indices, 'Title'] = new_titles

# Save the updated DataFrame back to CSV
pi_info_df.to_csv('PI_Info_Updated_Reporter7_largest_component.csv', index=False)
