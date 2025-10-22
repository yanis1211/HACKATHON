"""Fetch pharmacies from OpenStreetMap Overpass API and save raw + processed files.

This script queries Overpass for objects with amenity=pharmacy inside a bounding
box (default: metropolitan France). It writes a raw JSON and a cleaned CSV to
the project's data directory.

Usage:
    python -m src.etl.fetch_pharmacies
    or call fetch_and_save_pharmacies() from code.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_BBOX = {
    # left, bottom, right, top  (lon/lat)
    "left": -5.142222,
    "bottom": 41.333740,
    "right": 9.560000,
    "top": 51.089062,
}


def _overpass_query(bbox: Dict[str, float]) -> str:
    return (
        "[out:json][timeout:900];"
        "(node[\"amenity\"=\"pharmacy\"]({bottom},{left},{top},{right});"
        "way[\"amenity\"=\"pharmacy\"]({bottom},{left},{top},{right});"
        "relation[\"amenity\"=\"pharmacy\"]({bottom},{left},{top},{right});"
        ");out center;"
    ).format(**bbox)


def fetch_pharmacies(bbox: Dict[str, float] | None = None) -> dict:
    """Query Overpass API and return parsed JSON response."""
    if bbox is None:
        bbox = DEFAULT_BBOX
    url = "https://overpass-api.de/api/interpreter"
    q = _overpass_query(bbox)
    headers = {"User-Agent": "h4xathon-pharmacies-script/1.0 (email@example.com)"}
    resp = requests.post(url, data={"data": q}, headers=headers, timeout=300)
    resp.raise_for_status()
    return resp.json()


def _element_to_row(el: dict) -> dict:
    tags = el.get("tags", {})
    # lat/lon: nodes have lat/lon; ways/relation may have center
    if el.get("type") == "node":
        lat = el.get("lat")
        lon = el.get("lon")
    else:
        center = el.get("center") or {}
        lat = center.get("lat")
        lon = center.get("lon")

    return {
        "osm_id": el.get("id"),
        "osm_type": el.get("type"),
        "name": tags.get("name"),
        "lat": lat,
        "lon": lon,
        "street": tags.get("addr:street"),
        "postcode": tags.get("addr:postcode"),
        "city": tags.get("addr:city") or tags.get("addr:place"),
        "raw_tags": tags,
    }


def fetch_and_save_pharmacies(bbox: Dict[str, float] | None = None) -> None:
    data = fetch_pharmacies(bbox=bbox)
    raw_path = RAW_DIR / "pharmacies_overpass.json"
    with raw_path.open("w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    elements = data.get("elements", [])
    rows: List[Dict] = []
    for el in elements:
        row = _element_to_row(el)
        # skip items without coordinates
        if row["lat"] is None or row["lon"] is None:
            continue
        rows.append(row)

    # save CSV
    import csv

    processed_path = PROCESSED_DIR / "pharmacies.csv"
    fieldnames = ["osm_id", "osm_type", "name", "lat", "lon", "street", "postcode", "city"]
    with processed_path.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"Saved {len(rows)} pharmacies -> {processed_path}")


if __name__ == "__main__":
    print("Fetching pharmacies from Overpass (this may take a while)...")
    fetch_and_save_pharmacies()
