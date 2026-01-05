import time
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Add an email (recommended by NCBI). API key optional but helpful for higher rate limits.
NCBI_EMAIL = "your_email@example.com"   # <-- change this
NCBI_API_KEY = None                    # e.g. "xxxx" or keep None

HEADERS = {
    "User-Agent": f"NIH-network-affiliation-script/1.0 (contact: {NCBI_EMAIL})"
}

def _text(x):
    return x.text.strip() if x is not None and x.text else ""

def get_pi_affiliation_from_pubmed(pi_name: str, retmax: int = 10, timeout: int = 20) -> str:
    """
    Returns:
      - affiliation string (multiple joined by '; ')
      - 'No articles found.'
      - 'No valid affiliation found'
      - or an 'Error: ...' string
    Logic:
      1) search by full name
      2) pick most recent PMID (pub date sort)
      3) fetch record
      4) match author by last name ONLY
      5) collect affiliations, join by '; '
    """
    try:
        pi_last = pi_name.split(",")[0].strip().upper()

        params = {
            "db": "pubmed",
            "term": pi_name,
            "sort": "pub+date",
            "retmax": str(retmax),
            "retmode": "xml",
            "email": NCBI_EMAIL,
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        r = requests.get(ESEARCH, params=params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        root = ET.fromstring(r.text)

        pmids = [n.text for n in root.findall(".//IdList/Id") if n.text]
        if not pmids:
            return "No articles found."

        most_recent_pmid = pmids[0]

        params = {
            "db": "pubmed",
            "id": most_recent_pmid,
            "retmode": "xml",
            "email": NCBI_EMAIL,
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        r = requests.get(EFETCH, params=params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        root = ET.fromstring(r.text)

        affiliations = []
        matched = False

        # PubmedArticle/MedlineCitation/Article/AuthorList/Author
        for author in root.findall(".//AuthorList/Author"):
            last = _text(author.find("LastName")).upper()
            if not last:
                continue

            # ✅ ONLY match last name
            if last == pi_last:
                matched = True
                for aff in author.findall(".//AffiliationInfo/Affiliation"):
                    a = _text(aff)
                    if a:
                        affiliations.append(a)
                break  # stop at first last-name match

        # Deduplicate while preserving order
        seen = set()
        affiliations = [a for a in affiliations if not (a in seen or seen.add(a))]

        if not matched:
            return "No valid affiliation found"
        if not affiliations:
            return "No valid affiliation found"

        return "; ".join(affiliations)

    except Exception as e:
        return f"Error: {str(e)}"

def fetch_all_affiliations(pis, max_workers: int = 3, delay_per_call: float = 0.12):
    """
    Batch runner with light throttling.
    delay_per_call is applied after each completed PI to reduce rate-limit risk.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_pi = {ex.submit(get_pi_affiliation_from_pubmed, pi): pi for pi in pis}

        done = 0
        total = len(pis)

        for fut in as_completed(future_to_pi):
            pi = future_to_pi[fut]
            results[pi] = fut.result()
            done += 1

            if done % 25 == 0 or done == total:
                print(f"Processed {done}/{total}")

            # gentle throttling
            time.sleep(delay_per_call)

    return results

# -------------------
# RUN YOUR PIPELINE
# -------------------
a = pd.read_csv("data/PI_Info_missing_or_error.csv")
all_unique_pis_list = list(a["PI Name"])

aff_map = fetch_all_affiliations(all_unique_pis_list, max_workers=3, delay_per_call=0.12)

pi_info_df = pd.DataFrame({
    "PI Name": all_unique_pis_list,
    "Title": [aff_map.get(pi, "Error: missing result") for pi in all_unique_pis_list]
})

print(pi_info_df.head(20))

pi_info_df.to_csv("data/PI_Info_missing_or_error_updated.csv", index=False)



