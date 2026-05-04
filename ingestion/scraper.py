import requests
from bs4 import BeautifulSoup
import time
import logging
from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/kelly/Documents/sheria-intelligence/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class LegalDocument:
    title: str
    source_url: str
    domain: str
    doc_type: str
    raw_content: str
    version_date: str  # tracks which amendment version we scraped

# New Kenya Law site — versioned URLs, updated as amendments are published
TARGET_ACTS = [
    (
        "Employment Act 2007",
        "https://new.kenyalaw.org/akn/ke/act/2007/11/eng@2024-04-26",
        "employment",
        "2024-04-26"
    ),
    (
        "Labour Relations Act 2007",
        "https://new.kenyalaw.org/akn/ke/act/2007/14/eng@2022-12-31",
        "employment",
        "2022-12-31"
    ),
    (
        "Land Act 2012",
        "https://new.kenyalaw.org/akn/ke/act/2012/6/eng@2022-12-31",
        "land",
        "2022-12-31"
    ),
    (
        "Land Registration Act 2012",
        "https://new.kenyalaw.org/akn/ke/act/2012/3/eng@2022-12-31",
        "land",
        "2022-12-31"
    ),
    (
        "Companies Act 2015",
        "https://new.kenyalaw.org/akn/ke/act/2015/17/eng@2022-12-31",
        "business",
        "2022-12-31"
    ),
    (
        "Business Registration Service Act 2015",
        "https://new.kenyalaw.org/akn/ke/act/2015/15/eng@2022-12-31",
        "business",
        "2022-12-31"
    ),
]

class KenyaLawScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        })
        self.scraped = []
        self.failed = []

    def scrape_act(self, title: str, url: str, domain: str, version_date: str) -> Optional[LegalDocument]:
        try:
            logger.info(f"Scraping: {title}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # New Kenya Law renders content in <div class="akn-doc">
            content_div = (
                soup.find("div", class_="akn-doc") or
                soup.find("div", class_="akn-body") or
                soup.find("main") or
                soup.find("article")
            )

            if not content_div:
                logger.warning(f"No content div found for: {title}")
                return None

            raw_text = content_div.get_text(separator="\n", strip=True)

            if len(raw_text) < 500:
                logger.warning(f"Content too short for: {title} ({len(raw_text)} chars)")
                return None

            logger.info(f"✅ Scraped: {title} ({len(raw_text):,} chars)")

            return LegalDocument(
                title=title,
                source_url=url,
                domain=domain,
                doc_type="legislation",
                raw_content=raw_text,
                version_date=version_date
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed: {title} — {e}")
            self.failed.append(title)
            return None

    def scrape_all(self) -> list[LegalDocument]:
        logger.info(f"Starting scrape of {len(TARGET_ACTS)} acts...")

        for title, url, domain, version_date in TARGET_ACTS:
            doc = self.scrape_act(title, url, domain, version_date)
            if doc:
                self.scraped.append(doc)
            time.sleep(2)

        logger.info(f"\n📊 Scrape complete:")
        logger.info(f"   ✅ Success: {len(self.scraped)}")
        logger.info(f"   ❌ Failed:  {len(self.failed)}")
        if self.failed:
            logger.info(f"   Failed: {self.failed}")

        return self.scraped


if __name__ == "__main__":
    scraper = KenyaLawScraper()
    docs = scraper.scrape_all()

    for doc in docs:
        print(f"\n{'='*60}")
        print(f"Title:   {doc.title}")
        print(f"Domain:  {doc.domain}")
        print(f"Version: {doc.version_date}")
        print(f"Chars:   {len(doc.raw_content):,}")
        print(f"Preview: {doc.raw_content[:300]}...")
