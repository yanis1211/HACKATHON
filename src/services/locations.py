"""Location utilities: geocode addresses and find nearest pharmacies.

This module loads the processed `data/processed/pharmacies.csv` produced by
`src.etl.fetch_pharmacies` and exposes helpers to geocode addresses via
Nominatim and compute nearest pharmacies.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import math
import requests
import csv

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MANUAL_DIR = ROOT / "data" / "manual"
# prefer the manual CSV if provided (santé.fr), otherwise fallback to processed/pharmacies.csv
MANUAL_CSV = MANUAL_DIR / "santefr-lieux-vaccination-grippe-pharmacie.csv"
PHARMACIES_CSV = PROCESSED_DIR / "pharmacies.csv"


@dataclass
class Pharmacy:
    osm_id: str
    name: Optional[str]
    lat: float
    lon: float
    street: Optional[str]
    postcode: Optional[str]
    city: Optional[str]


def load_pharmacies() -> List[Pharmacy]:
    # choose source: manual CSV (semi-colon) has coordinates columns Adresse_latitude/Adresse_longitude
    if MANUAL_CSV.exists():
        rows: List[Pharmacy] = []
        with MANUAL_CSV.open("r", encoding="utf8") as f:
            # file uses semicolon separator and may have quoted fields
            reader = csv.DictReader(f, delimiter=";")
            for r in reader:
                try:
                    # some lat/lon values may have whitespace
                    lat_s = (r.get("Adresse_latitude") or r.get("Adresse_latitude ") or "").strip()
                    lon_s = (r.get("Adresse_longitude") or r.get("Adresse_longitude ") or "").strip()
                    lat = float(lat_s)
                    lon = float(lon_s)
                except Exception:
                    continue
                name = (r.get("Titre") or r.get("Nom") or "").strip().strip('"')
                street = (r.get("Adresse_voie 1") or r.get("Adresse_voie 1 ") or r.get("Adresse_voie") or "").strip().strip('"')
                postcode = (r.get("Adresse_codepostal") or r.get("Adresse_codepostal ") or "").strip()
                city = (r.get("Adresse_ville") or "").strip().strip('"')
                finess = (r.get("Finess") or "").strip()
                osm_id = finess if finess else ""
                rows.append(
                    Pharmacy(
                        osm_id=osm_id,
                        name=name,
                        lat=lat,
                        lon=lon,
                        street=street,
                        postcode=postcode,
                        city=city,
                    )
                )
        return rows

    # fallback to processed CSV
    if not PHARMACIES_CSV.exists():
        raise FileNotFoundError(f"neither {MANUAL_CSV} nor {PHARMACIES_CSV} found — provide a CSV or run src.etl.fetch_pharmacies")
    rows = []
    with PHARMACIES_CSV.open("r", encoding="utf8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                lat = float(r["lat"])
                lon = float(r["lon"])
            except Exception:
                continue
            rows.append(
                Pharmacy(
                    osm_id=str(r.get("osm_id") or ""),
                    name=r.get("name"),
                    lat=lat,
                    lon=lon,
                    street=r.get("street"),
                    postcode=r.get("postcode"),
                    city=r.get("city"),
                )
            )
    return rows


def geocode_address(address: str) -> Optional[dict]:
    """Geocode an address using Nominatim. Returns dict with 'lat' and 'lon'."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {"User-Agent": "h4xathon-geocode/1.0 (email@example.com)"}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    return data[0]


def _haversine_meters(lat1, lon1, lat2, lon2):
    # returns distance in meters
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_pharmacies(lat: float, lon: float, n: int = 5) -> List[dict]:
    pharmacies = load_pharmacies()
    scored = []
    for p in pharmacies:
        d = _haversine_meters(lat, lon, p.lat, p.lon)
        scored.append((d, p))
    scored.sort(key=lambda x: x[0])
    out = []
    for d, p in scored[:n]:
        out.append({
            "osm_id": p.osm_id,
            "name": p.name,
            "lat": p.lat,
            "lon": p.lon,
            "street": p.street,
            "postcode": p.postcode,
            "city": p.city,
            "distance_m": d,
        })
    return out
