import random
from concurrent.futures import ThreadPoolExecutor
import json
import re
from unidecode import unidecode
import wikipedia

def _clean_text(text: str) -> str:
    """
    Cleans text by unidecoding, removing line breaks, and multiple whitespaces.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = unidecode(text)
    text = re.sub(r'[\n\t]', '', text)
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'\\+', '', text)
    return text

def _get_wikipedia_data(title: str) -> tuple[dict[str, str], list[str]]:
    """
    Fetches data from a wikipedia page.

    Args:
        title (str): The title of the wikipedia page to retrieve data from.

    Returns:
        tuple[dict[str, str], list[str]]: A tuple where the first entry contains the title and the 
            summary of the page and the second entry contains all links of the page.
    """
    try:
        page = wikipedia.page(title)

        page_links = page.links

        data = {
            'title': _clean_text(page.title),
            'summary': _clean_text(page.summary)
        }

        return data, page_links
    except:
        return None, []

def scrape_wikipedia(start_page: str, n_pages: str, sample_size: int = 100) -> list[dict[str, str]]:
    """
    Function for scraping data from wikipedia. The scraper starts on a page and starts scraping linked pages.

    Args:
        start_page (str): The page the scraper will start with.
        n_pages (int): The number of pages to be scraped.
        sample_size (int): The number of pages that are scraped at the same time.

    Returns:
        list[dict[str, str]]: A list containing the scraped data.
    """
    all_links: set[str] = {start_page}
    data: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=sample_size) as executor:
        while len(data) < n_pages and len(all_links) > 0:
            current_sample_size = min(sample_size, len(all_links), n_pages - len(data))

            link_sample = random.sample(list(all_links), current_sample_size)

            for link in link_sample:
                all_links.remove(link)

            futures = [executor.submit(_get_wikipedia_data, link) for link in link_sample]
            for future in futures:
                page_data, links = future.result()
                if page_data:
                    data.append(page_data)

                    all_links.update(links)
            
            print(f"Fetched {len(data)}/{n_pages} pages.")

    data = list({page['title']: page for page in data}.values())

    return data

if __name__ == '__main__':
    data: list[dict[str, str]] = scrape_wikipedia('Deep Learning', 10000, 250)

    json_data = json.dumps(data)

    with open('./datasets/data.json', 'w') as f:
        f.write(json_data)