
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


BASE_URL = "https://www.bigbrainss.com/"
DOMAIN = "www.bigbrainss.com"
MAX_PAGES = 40
DELAY = 1.0

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "knowledge_base.json"
BACKUP_FILE = DATA_DIR / "knowledge_base_backup.json"

USER_AGENT = "BigBrainsCustomerSupportAgent/1.0"

SKIP_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
    ".pdf", ".zip", ".mp4", ".mp3", ".css", ".js",
    ".ico", ".woff", ".woff2", ".ttf",
)

# Public pages that are not useful as customer-support knowledge.
SKIP_PATH_PREFIXES = (
    "/register",
    "/login",
    "/logout",
    "/forgot-password",
    "/userprofile",
    "/reset-password",
)

# Remove repeated promotional blocks when they appear in main content.
PROMOTIONAL_PHRASES = (
    "build your portfolio in minutes",
    "show what you're to companies",
)


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_url(url):
    url, _ = urldefrag(url)
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return None

    if parsed.netloc.lower() not in (
        "www.bigbrainss.com",
        "bigbrainss.com",
    ):
        return None

    path = parsed.path or "/"

    if path.lower().endswith(SKIP_EXTENSIONS):
        return None

    # Skip authentication and personal-profile pages.
    if path.lower().startswith(SKIP_PATH_PREFIXES):
        return None

    # Remove query strings and standardize the domain.
    return "https://www.bigbrainss.com" + path


def get_category(url, title):
    path = urlparse(url).path.lower()
    title = title.lower()

    # Specific routes first: this prevents "AI" or "service"
    # in a title from overriding the correct category.
    if any(word in path or word in title for word in (
        "privacy", "policy", "refund", "cancellation",
        "terms", "faq", "delete-account",
    )):
        return "policies"

    if "/internships" in path or "/careers" in path:
        return "internships"

    if "/coursedetails" in path or path.rstrip("/") == "/courses":
        return "courses"

    if any(word in path for word in (
        "/notes", "/workshops", "/blog", "/resources",
    )):
        return "learning_resources"

    if any(word in path for word in (
        "/pricing", "/products", "/services",
    )):
        return "products_services"

    if any(word in path for word in (
        "/about-us", "/contact", "/ownership-statement",
    )):
        return "about"

    if path == "/" or "learn anytime" in title:
        return "learning_resources"

    return "general"


def load_robots():
    try:
        response = requests.get(
            urljoin(BASE_URL, "robots.txt"),
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )

        if response.status_code != 200:
            print("Cannot verify robots.txt:", response.status_code)
            return None

        parser = RobotFileParser()
        parser.parse(response.text.splitlines())
        return parser

    except requests.RequestException as error:
        print("robots.txt error:", error)
        return None


def extract_main_content(soup):
    # Remove elements that are normally not page content.
    for element in soup.select(
        "script, style, noscript, svg, nav, footer, "
        "header, form, button, iframe"
    ):
        element.decompose()

    # Ignore common repeated promotional or overlay components.
    for element in soup.select(
        '[class*="cookie"], [id*="cookie"], '
        '[class*="modal"], [id*="modal"], '
        '[class*="popup"], [id*="popup"], '
        '[class*="breadcrumb"], [class*="marquee"], '
        '[class*="ticker"], [class*="sidebar"]'
    ):
        element.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    # Extract meaningful text blocks in reading order.
    blocks = main.select(
        "h1, h2, h3, h4, p, li, dt, dd, blockquote"
    )

    if not blocks:
        blocks = [main]

    cleaned_blocks = []
    seen_blocks = set()

    for block in blocks:
        text = clean_text(block.get_text(" ", strip=True))

        if len(text) < 3:
            continue

        normalized = text.casefold()

        if normalized in seen_blocks:
            continue

        # Remove repeated portfolio marketing banner text.
        if any(phrase in normalized for phrase in PROMOTIONAL_PHRASES):
            continue

        seen_blocks.add(normalized)
        cleaned_blocks.append(text)

    # Fall back to full main text if block extraction was too sparse.
    content = " ".join(cleaned_blocks)

    if len(content) < 80:
        content = clean_text(main.get_text(" ", strip=True))

    return clean_text(content)


def extract_rendered_page(page, url):
    response = page.goto(
        url,
        wait_until="domcontentloaded",
        timeout=30000,
    )

    if response is None:
        print("No HTTP response:", url)
        return None, []

    if response.status >= 400:
        print("HTTP", response.status, "Skipping:", url)
        return None, []

    # Give client-side content a chance to render.
    page.wait_for_timeout(2000)

    html = page.content()
    soup = BeautifulSoup(html, "html.parser")

    title = (
        clean_text(soup.title.get_text(" "))
        if soup.title
        else url
    )

    content = extract_main_content(soup)

    # Collect links from the rendered page.
    raw_links = page.locator("a[href]").evaluate_all(
        "(els) => els.map(el => el.href)"
    )

    links = []

    for raw_link in raw_links:
        normalized = normalize_url(urljoin(url, raw_link))

        if normalized and normalized not in links:
            links.append(normalized)

    # Avoid saving blank pages and pages with almost no real content.
    if len(content) < 80:
        print("Insufficient readable content:", url)
        return None, links

    record = {
        "title": title,
        "url": url,
        "category": get_category(url, title),
        "content": content,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }

    return record, links


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve the previous knowledge base before replacing it.
    if OUTPUT_FILE.exists():
        BACKUP_FILE.write_bytes(OUTPUT_FILE.read_bytes())
        print("Previous knowledge base backed up to:", BACKUP_FILE)

    robots = load_robots()

    if robots is None:
        print("Stopped: crawling rules could not be verified.")
        return

    seed_paths = [
        "/",
        "/courses",
        "/pricing",
        "/privacyPolicy",
        "/faq",
        "/internships",
        "/terms",
        "/refund",
        "/about-us",
        "/contact",
        "/notes",
        "/workshops",
        "/blog",
        "/careers",
    ]

    queue = deque()

    for path in seed_paths:
        url = normalize_url(urljoin(BASE_URL, path))

        if url and url not in queue:
            queue.append(url)

    queued = set(queue)
    visited = set()
    records = []
    failures = []

    print("Starting improved Big Brains scraper...")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)

        context = browser.new_context(
            user_agent=USER_AGENT,
            ignore_https_errors=False,
        )

        page = context.new_page()

        while queue and len(visited) < MAX_PAGES:
            url = queue.popleft()

            if url in visited:
                continue

            visited.add(url)

            if not robots.can_fetch(USER_AGENT, url):
                print("Disallowed by robots.txt:", url)
                continue

            print("\nCrawling:", url)

            try:
                record, links = extract_rendered_page(page, url)

                if record:
                    records.append(record)
                    print("Saved:", record["title"])
                    print("Category:", record["category"])
                    print("Content characters:", len(record["content"]))

                for link in links:
                    if (
                        link not in queued
                        and link not in visited
                        and len(visited) + len(queue) < MAX_PAGES
                    ):
                        queue.append(link)
                        queued.add(link)

            except Exception as error:
                print("Page failed:", url, error)
                failures.append({
                    "url": url,
                    "error": str(error),
                })

            time.sleep(DELAY)

        browser.close()

    # Remove exact duplicate page content.
    unique_records = []
    seen_content = set()

    for record in records:
        fingerprint = record["content"].casefold()

        if fingerprint in seen_content:
            print("Duplicate content skipped:", record["url"])
            continue

        seen_content.add(fingerprint)
        unique_records.append(record)

    category_counts = {}

    for record in unique_records:
        category = record["category"]
        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

    result = {
        "website": BASE_URL,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "pages_attempted": len(visited),
        "pages_saved": len(unique_records),
        "failed_pages": failures,
        "category_counts": category_counts,
        "documents": unique_records,
    }

    OUTPUT_FILE.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n========== SCRAPING SUMMARY ==========")
    print("Pages attempted:", len(visited))
    print("Pages saved:", len(unique_records))
    print("Failed pages:", len(failures))
    print("Categories:", category_counts)
    print("Output:", OUTPUT_FILE)

    if not unique_records:
        print("WARNING: No useful content was collected.")


if __name__ == "__main__":
    main()