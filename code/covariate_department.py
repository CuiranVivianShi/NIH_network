from concurrent.futures import ThreadPoolExecutor
from read_data import get_unique_pis
import requests
from bs4 import BeautifulSoup
import urllib.parse
import pandas as pd


all_unique_pis_list = list(get_unique_pis())


def get_pi_department(pi_name):
    base_url = "https://pubmed.ncbi.nlm.nih.gov"
    query = urllib.parse.quote_plus(pi_name)
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
                if pi_name.split(',')[0].lower() == last_name.lower():
                    affiliation_link = author.find('a', class_='affiliation-link')
                    if affiliation_link:
                        return affiliation_link.get('title')
            return "No valid affiliation found"
    except Exception as e:
        return f"Error: {str(e)}"


def fetch_all_departments(pis):
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(get_pi_department, pis))
    return results


# Prepare to store data
pi_data = []

# Fetch all PI department data
results = fetch_all_departments(all_unique_pis_list)

for pi, title in zip(all_unique_pis_list, results):
    pi_data.append({'PI Name': pi, 'Title': title})

# Create a DataFrame from the collected data
pi_info_df = pd.DataFrame(pi_data)
print(pi_info_df.head(20))

# Optionally, save to CSV
pi_info_df.to_csv('PI_Info.csv', index=False)









