from read_data import get_unique_pis
import requests
from bs4 import BeautifulSoup

all_unique_pis_list = list(get_unique_pis())

def fetch_results_for_all_pages(base_url, term):
    all_results = []
    current_page = 1
    latest_year = 0
    latest_article_link = None

    while True:
        url = f"{base_url}?term={term}&page={current_page}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        articles = soup.find_all('div', class_='docsum-wrap')

        for article in articles:
            citation = article.find('span', class_='docsum-journal-citation full-journal-citation')
            if citation:
                year = max([int(s) for s in citation.text.split() if s.isdigit()], default=0)
                if year > latest_year:
                    latest_year = year
                    link_tag = article.find('a', class_='docsum-title')
                    if link_tag and link_tag.has_attr('href'):
                        latest_article_link = f"https://pubmed.ncbi.nlm.nih.gov{link_tag['href']}"

        # Check if there are more pages
        page_input = soup.find('input', {'id': 'page-number-input'})
        if not page_input or current_page >= int(page_input.get('max')):
            break
        current_page += 1

    return latest_article_link

# Variables
base_url = "https://pubmed.ncbi.nlm.nih.gov"
#query = "JATLOW%2C+PETER+I"
query = "RICHERSON%2C+GEORGE+B"
latest_article_link = fetch_results_for_all_pages(base_url, query)
print("Latest Article Link:", latest_article_link)



response1 = requests.get(latest_article_link)
soup1 = BeautifulSoup(response1.text, 'html.parser')

authors = soup1.find_all('span', class_='authors-list-item')

#author_name_to_find = "PETER I JATLOW"
author_name_to_find = 'RICHERSON, GEORGE B'

author_name_to_find = 'RICHERSON, GEORGE B'   # Only the last name


# Loop through each author
for author in authors:
    author_name_element = author.find('a', class_='full-name')
    if author_name_element:
        last_name = author_name_element.text.split()[-1]

        if author_name_to_find.split(',')[0].lower() == last_name.lower():
            affiliation_link = author.find('a', class_='affiliation-link')
            if affiliation_link:
                title = affiliation_link.get('title')
                print(f"Title for {author_name_to_find}: {title}")
                break
else:
    print(f"No affiliation title found for {author_name_to_find}.")

















