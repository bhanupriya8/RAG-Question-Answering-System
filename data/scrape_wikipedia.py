"""
Scrapes text summaries + a thumbnail image for each landmark from Wikipedia's
public REST API. No API key required.

Run this FIRST, on your laptop (needs internet access):
    python data/scrape_wikipedia.py

Output:
    data/knowledge_base.json   -> list of {title, extract, image_path, url}
    data/images/<title>.jpg    -> downloaded thumbnail per landmark
"""
"""
Scrapes text summaries + a thumbnail image for each landmark from Wikipedia's
public REST API. No API key required.

Run this FIRST, on your laptop:
    python scrape_wikipedia.py

Output:
    knowledge_base.json       -> list of {title, extract, image_path, url}
    images/<title>.jpg        -> downloaded thumbnail per landmark
"""


"""
Scrapes text summaries + a thumbnail image for each landmark from Wikipedia's
public REST API. No API key required.

Run:
    python scrape_wikipedia.py

Output:
    knowledge_base.json
    images/<title>.jpg
"""

import json
import os
import re
import time
import requests

from landmark_list import LANDMARKS


WIKI_SUMMARY_URL = (
    "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(OUT_DIR, "images")
KB_PATH = os.path.join(OUT_DIR, "knowledge_base.json")


# Proper headers for Wikipedia/Wikimedia
HEADERS = {
    "User-Agent": (
        "ProMSA-Lite/1.0 "
        "(student project; Python requests)"
    ),
    "Accept": "*/*",
    "Referer": "https://en.wikipedia.org/",
}


def safe_filename(title: str) -> str:
    return re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        title
    ) + ".jpg"


def fetch_landmark(title: str):

    # ---------------------------------------------------------
    # Fetch Wikipedia summary
    # ---------------------------------------------------------

    url = WIKI_SUMMARY_URL.format(
        title.replace(" ", "_")
    )

    resp = None

    for attempt in range(4):

        try:

            resp = requests.get(
                url,
                headers=HEADERS,
                timeout=20
            )

        except requests.RequestException as e:

            print(
                f"  [skip] {title} -> request failed: {e}"
            )

            return None

        if resp.status_code == 200:
            break

        if resp.status_code == 429:

            wait = 5 * (attempt + 1)

            print(
                f"  [retry] {title} -> HTTP 429, "
                f"waiting {wait}s..."
            )

            time.sleep(wait)

        else:

            print(
                f"  [skip] {title} -> "
                f"summary HTTP {resp.status_code}"
            )

            return None

    if resp is None or resp.status_code != 200:

        print(
            f"  [skip] {title} -> "
            "could not fetch Wikipedia summary"
        )

        return None

    # ---------------------------------------------------------
    # Parse JSON
    # ---------------------------------------------------------

    try:

        data = resp.json()

    except ValueError:

        print(
            f"  [skip] {title} -> invalid JSON response"
        )

        return None

    extract = data.get(
        "extract",
        ""
    )

    if not extract:

        print(
            f"  [skip] {title} -> "
            f"no extract found "
            f"(page_type={data.get('type')})"
        )

        return None

    # ---------------------------------------------------------
    # Get thumbnail directly from Wikipedia API
    # ---------------------------------------------------------

    thumb = (
        data
        .get("thumbnail", {})
        .get("source")
    )

    image_path = None

    if not thumb:

        print(
            f"  [warn] {title} -> "
            "no thumbnail available from Wikipedia"
        )

    else:

        print(
            f"  [image] Downloading thumbnail..."
        )

        img_resp = None

        # -----------------------------------------------------
        # Download the EXACT URL returned by Wikipedia
        # Do NOT replace it with 800px
        # -----------------------------------------------------

        for attempt in range(4):

            try:

                img_resp = requests.get(
                    thumb,
                    headers=HEADERS,
                    timeout=30,
                    allow_redirects=True
                )

            except requests.RequestException as e:

                print(
                    f"  [retry] {title} -> "
                    f"image request failed: {e}"
                )

                time.sleep(3)
                continue

            # -------------------------------------------------
            # Successful image
            # -------------------------------------------------

            if (
                img_resp.status_code == 200
                and len(img_resp.content) > 500
            ):

                break

            # -------------------------------------------------
            # Rate limited
            # -------------------------------------------------

            if img_resp.status_code == 429:

                wait = 5 * (attempt + 1)

                print(
                    f"  [retry] {title} -> "
                    f"image HTTP 429, waiting {wait}s..."
                )

                time.sleep(wait)
                continue

            # -------------------------------------------------
            # Other error
            # -------------------------------------------------

            print(
                f"  [warn] {title} -> "
                f"image HTTP {img_resp.status_code}"
            )

            break

        # -----------------------------------------------------
        # Save image
        # -----------------------------------------------------

        if (
            img_resp is not None
            and img_resp.status_code == 200
            and len(img_resp.content) > 500
        ):

            fname = safe_filename(title)

            image_path = os.path.join(
                "images",
                fname
            )

            full_path = os.path.join(
                OUT_DIR,
                image_path
            )

            try:

                with open(
                    full_path,
                    "wb"
                ) as f:

                    f.write(img_resp.content)

                print(
                    f"  [ok] Image saved: {fname}"
                )

            except OSError as e:

                print(
                    f"  [warn] {title} -> "
                    f"could not save image: {e}"
                )

                image_path = None

        elif img_resp is not None:

            print(
                f"  [warn] {title} -> "
                f"image download failed "
                f"(HTTP {img_resp.status_code})"
            )

    # ---------------------------------------------------------
    # Return data
    # ---------------------------------------------------------

    return {
        "title": title,
        "extract": extract,
        "image_path": image_path,
        "url": (
            data
            .get("content_urls", {})
            .get("desktop", {})
            .get("page")
        ),
    }


def main():

    os.makedirs(
        IMG_DIR,
        exist_ok=True
    )

    kb = []

    # ---------------------------------------------------------
    # Process landmarks
    # ---------------------------------------------------------

    for i, title in enumerate(
        LANDMARKS,
        1
    ):

        print(
            f"[{i}/{len(LANDMARKS)}] Fetching: {title}"
        )

        entry = fetch_landmark(title)

        if entry:

            kb.append(entry)

        # Wait between Wikipedia requests
        time.sleep(1.5)

    # ---------------------------------------------------------
    # Save knowledge base
    # ---------------------------------------------------------

    with open(
        KB_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            kb,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ---------------------------------------------------------
    # Count images
    # ---------------------------------------------------------

    n_with_images = sum(
        1
        for entry in kb
        if entry["image_path"]
    )

    print()
    print(
        f"Done. {len(kb)} entries saved to "
        f"{KB_PATH}"
    )

    print(
        f"{n_with_images} / {len(kb)} entries have images "
        f"in {IMG_DIR}"
    )

    if n_with_images == 0:

        print()
        print(
            "WARNING: zero images downloaded."
        )


if __name__ == "__main__":
    main()