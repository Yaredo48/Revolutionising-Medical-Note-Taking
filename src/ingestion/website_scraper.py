import os
import requests
from bs4 import BeautifulSoup

def scrape_website(url, output_folder="data/raw/processed_websites", filename=None):
    os.makedirs(output_folder, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch {url}")
        print("Error:", e)
        return

    soup = BeautifulSoup(response.text, "lxml")
    text = soup.get_text(separator="\n")

    if not filename:
        filename = url.replace("https://", "").replace("http://", "").replace("/", "_") + ".txt"

    out_file = os.path.join(output_folder, filename)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Scraped {url} → {out_file}")

if __name__ == "__main__":
    test_url ="https://www.cdc.gov/flu/professionals/"
    scrape_website(test_url)
