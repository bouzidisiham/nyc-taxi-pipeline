# src/serve_app.py
from __future__ import annotations

import os
import time
import logging
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable

import streamlit as st
import pandas as pd
import pydeck as pdk
import joblib

# =========================================================
# LOGGING
# =========================================================
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("nyc-app")

# =========================================================
# CONFIG
# =========================================================
@dataclass(frozen=True)
class AppConfig:
    title: str = "NYC Yellow Taxi — Travel Time (Over-Engineered)"
    page_icon: str = "🚕"
    map_style: str = os.getenv(
        "MAP_STYLE",
        "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
    )
    lat_min: float = 40.50
    lat_max: float = 40.92
    lon_min: float = -74.27
    lon_max: float = -73.68
    model_candidates: Tuple[str, ...] = ("artifacts/model_rf.pkl", "artifacts/model.pkl")
    timeout_geocode: int = 6
    geocode_rate_min_delay: float = 1.0   # Nominatim politeness
    geocode_max_retries: int = 2
    geocode_backoff_base: float = 1.6
    circuit_breaker_threshold: int = 3
    circuit_breaker_cooldown_s: int = 30
    default_pickup: str = "Times Square, New York, NY"
    default_drop: str = "Washington Square Park, New York, NY"

CONFIG = AppConfig()

# =========================================================
# STREAMLIT PAGE CONFIG & STYLE
# =========================================================
st.set_page_config(page_title=CONFIG.title, page_icon=CONFIG.page_icon, layout="wide")

st.markdown("""
<style>
h1 span.title {
  background: linear-gradient(90deg,#0ea5e9 0%,#6366f1 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
div.card{padding:1rem 1.25rem;border-radius:18px;border:1px solid rgba(0,0,0,.07);
background:rgba(255,255,255,.8);box-shadow:0 8px 24px rgba(0,0,0,.06)}
.badge{display:inline-block;padding:.25rem .6rem;border-radius:999px;font-size:.78rem;
font-weight:600;border:1px solid rgba(0,0,0,.08)}
.badge.ok{background:#ecfdf5;color:#065f46}
.badge.warn{background:#fff7ed;color:#9a3412}
.badge.info{background:#eff6ff;color:#1e3a8a}
hr.soft{border:none;border-top:1px solid rgba(0,0,0,.08);margin:.75rem 0 1.25rem}
</style>
""", unsafe_allow_html=True)
st.markdown("<h1><span class='title'>NYC Yellow Taxi — Travel Time</span></h1>", unsafe_allow_html=True)

# =========================================================
# UTILITIES
# =========================================================
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def in_bounds(lat: float, lon: float) -> bool:
    return CONFIG.lat_min <= lat <= CONFIG.lat_max and CONFIG.lon_min <= lon <= CONFIG.lon_max

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    from math import radians, sin, cos, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return float(2*R*asin(sqrt(a)))

# =========================================================
# CIRCUIT BREAKER (pour géocodage)
# =========================================================
class CircuitBreaker:
    def __init__(self, threshold: int, cooldown_s: int):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.failures = 0
        self.open_until = 0.0

    def allow(self) -> bool:
        now = time.time()
        return now >= self.open_until

    def record_success(self):
        self.failures = 0
        self.open_until = 0.0

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.threshold:
            self.open_until = time.time() + self.cooldown_s
            log.warning("Circuit breaker OPEN for %ss", self.cooldown_s)

breaker = CircuitBreaker(CONFIG.circuit_breaker_threshold, CONFIG.circuit_breaker_cooldown_s)

# =========================================================
# GEOCODER STRATEGY + BACKOFF + CACHE
# =========================================================
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

FALLBACK_POI: Dict[str, Tuple[float, float]] = {
    "times square, new york, ny": (40.758056, -73.985556),
    "washington square park, new york, ny": (40.730823, -73.997332),
    "20 w 34th st, new york, ny 10001": (40.748441, -73.985664),
    "180 greenwich st, new york, ny 10007": (40.711542, -74.012308),
    "200 central park west, new york, ny 10024": (40.781324, -73.973988),
    "11 wall st, new york, ny 10005": (40.707565, -74.011938),
}

@st.cache_resource
def get_nominatim() -> Nominatim:
    g = Nominatim(user_agent="siham-nyc-taxi-app/2.0-over", timeout=CONFIG.timeout_geocode, scheme="https")
    g.geocode = RateLimiter(g.geocode, min_delay_seconds=CONFIG.geocode_rate_min_delay)
    return g

def normalize_address(addr: str) -> str:
    a = (addr or "").strip()
    if not a:
        return a
    if "new york" not in a.lower():
        a = f"{a}, New York, NY"
    return a

def backoff_retry(fn: Callable[..., Any], max_retries: int, base: float) -> Callable[..., Any]:
    """Wrap a callable with retry + exponential backoff on geopy exceptions."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        last_exc = None
        for i in range(max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                last_exc = e
                delay = (base ** i)
                log.warning("Geocode attempt %s failed: %s — retry in %.1fs", i+1, e, delay)
                time.sleep(delay)
        if last_exc:
            raise last_exc
    return wrapper

@st.cache_data(show_spinner=False, ttl=60*10)
def geocode_cached(addr_norm: str) -> Optional[Tuple[float, float, str]]:
    """
    Appel Nominatim en 2 passes (bounded puis libre) + clamp bounds.
    Cache courte durée.
    """
    if not addr_norm:
        return None
    g = get_nominatim()

    def pass_strict() -> Optional[Tuple[float, float, str]]:
        loc = g.geocode(
            addr_norm, addressdetails=False, limit=1,
            viewbox=[(CONFIG.lon_min, CONFIG.lat_min), (CONFIG.lon_max, CONFIG.lat_max)],
            bounded=True
        )
        if loc:
            return float(loc.latitude), float(loc.longitude), (loc.address or addr_norm)
        return None

    def pass_soft() -> Optional[Tuple[float, float, str]]:
        loc = g.geocode(addr_norm, addressdetails=False, limit=1)
        if loc:
            return float(loc.latitude), float(loc.longitude), (loc.address or addr_norm)
        return None

    # wrap avec backoff
    strict = backoff_retry(pass_strict, CONFIG.geocode_max_retries, CONFIG.geocode_backoff_base)
    soft   = backoff_retry(pass_soft,   CONFIG.geocode_max_retries, CONFIG.geocode_backoff_base)

    try:
        if not breaker.allow():
            log.warning("Circuit open: using fallback table only.")
            raise GeocoderServiceError("circuit_open")

        res = strict()
        if not res:
            res = soft()
        if res:
            lat, lon, label = res
            lat = clamp(lat, CONFIG.lat_min, CONFIG.lat_max)
            lon = clamp(lon, CONFIG.lon_min, CONFIG.lon_max)
            breaker.record_success()
            return (lat, lon, label)

    except (GeocoderTimedOut, GeocoderServiceError) as e:
        breaker.record_failure()
        log.error("Geocode error: %s", e)

    # fallback local
    key = addr_norm.lower()
    if key in FALLBACK_POI:
        plat, plon = FALLBACK_POI[key]
        return (plat, plon, addr_norm)

    return None

def geocode_address(addr: str) -> Tuple[Optional[float], Optional[float], Optional[str], str]:
    """
    Renvoie (lat, lon, label, source) où source ∈ {"nominatim","fallback","none"}.
    """
    addr_norm = normalize_address(addr)
    hit = geocode_cached(addr_norm)
    if hit:
        lat, lon, label = hit
        source = "fallback" if addr_norm.lower() in FALLBACK_POI else "nominatim"
        return lat, lon, label, source
    return None, None, None, "none"

# =========================================================
# MODEL LOADER
# =========================================================
class ModelNotFound(Exception): ...

@st.cache_resource(show_spinner=False)
def load_model() -> Any:
    for path in CONFIG.model_candidates:
        p = Path(path)
        if p.exists() and p.is_file():
            try:
                m = joblib.load(p)
                log.info("Loaded model: %s", p)
                return m
            except Exception as e:
                log.exception("Failed to load model %s: %s", p, e)
    raise ModelNotFound(f"No model found in {CONFIG.model_candidates}")

# =========================================================
# FEATURE BUILDER
# =========================================================
@dataclass
class InferenceContext:
    pickup_address: str
    drop_address: str
    pickup_hour: int
    pickup_dow: int
    pickup_month: int
    passenger_count: int
    vendor: int
    ratecode: int
    payment_type: int

@dataclass
class FeatureBuilder:
    ctx: InferenceContext
    pickup_lat: float
    pickup_lon: float
    drop_lat: float
    drop_lon: float

    def build(self) -> pd.DataFrame:
        rush_hour = int(self.ctx.pickup_hour in [7,8,9,16,17,18])
        is_weekend = int(self.ctx.pickup_dow >= 5)
        hav_km = haversine_km(self.pickup_lat, self.pickup_lon, self.drop_lat, self.drop_lon)
        man_km = hav_km  # placeholder
        bearing = 0.0   # placeholder

        row = {
            "VendorID": self.ctx.vendor,
            "RatecodeID": self.ctx.ratecode,
            "payment_type": self.ctx.payment_type,
            "is_weekend": is_weekend,
            "rush_hour": rush_hour,
            "pickup_hour": self.ctx.pickup_hour,
            "pickup_dow": self.ctx.pickup_dow,
            "pickup_month": self.ctx.pickup_month,
            "passenger_count": self.ctx.passenger_count,
            "trip_distance": hav_km,
            "haversine_km": hav_km,
            "manhattan_km": man_km,
            "bearing_deg": bearing,
            "congestion_surcharge": 0.0,
            "mta_tax": 0.0,
            "tolls_amount": 0.0,
            "improvement_surcharge": 0.0,
            "fare_amount": 0.0,
            "extra": 0.0,
            "h3_pickup_r8": None,
            "h3_dropoff_r8": None,
        }
        return pd.DataFrame([row])

# =========================================================
# UI — LEFT: inputs / RIGHT: results
# =========================================================
left, right = st.columns([1.05, 1.35], gap="large")

with left:
    st.subheader("Paramètres")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            pickup_address = st.text_input(
                "Adresse départ (pickup)",
                value=CONFIG.default_pickup,
                placeholder="Ex: 20 W 34th St, New York, NY 10001"
            )
        with c2:
            drop_address = st.text_input(
                "Adresse arrivée (dropoff)",
                value=CONFIG.default_drop,
                placeholder="Ex: 180 Greenwich St, New York, NY 10007"
            )

        st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            pickup_hour = st.slider("Pickup Hour", 0, 23, 8)
            pickup_dow = st.slider("Pickup Day of Week (Mon=0)", 0, 6, 2)
            pickup_month = st.slider("Pickup Month", 1, 12, 1)
        with c4:
            passenger_count = st.number_input("Passenger Count", 1, 6, 1)
            vendor = st.selectbox("VendorID", [1,2], index=0)
            ratecode = st.selectbox("RatecodeID", [1,2,3,4,5,6], index=0)
            payment_type = st.selectbox("Payment Type", [1,2,3,4], index=0)

        st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
        btn_predict = st.button("Prédire la durée (minutes)", type="primary", use_container_width=True)
with right:
    st.subheader("📈 Résultats & Insights")

    if btn_predict:
        with st.spinner("Géocodage…"):
            plat, plon, plabel, psrc = geocode_address(pickup_address)
            dlat, dlon, dlabel, dsrc = geocode_address(drop_address)

        if plat is None or plon is None:
            st.error("Adresse départ introuvable dans NYC. Essaie une autre formulation.")
            st.stop()
        if dlat is None or dlon is None:
            st.error("Adresse arrivée introuvable dans NYC. Essaie une autre formulation.")
            st.stop()

        # bornage défensif
        if not (in_bounds(plat, plon) and in_bounds(dlat, dlon)):
            st.warning("Géocodé en dehors de la boîte NYC — coordinates clampées.")
            plat, plon = clamp(plat, CONFIG.lat_min, CONFIG.lat_max), clamp(plon, CONFIG.lon_min, CONFIG.lon_max)
            dlat, dlon = clamp(dlat, CONFIG.lat_min, CONFIG.lat_max), clamp(dlon, CONFIG.lon_min, CONFIG.lon_max)

        ctx = InferenceContext(
            pickup_address=pickup_address, drop_address=drop_address,
            pickup_hour=pickup_hour, pickup_dow=pickup_dow, pickup_month=pickup_month,
            passenger_count=passenger_count, vendor=vendor, ratecode=ratecode, payment_type=payment_type
        )
        feats = FeatureBuilder(ctx, plat, plon, dlat, dlon).build()

        try:
            model = load_model()
        except ModelNotFound as e:
            st.error(str(e))
            st.stop()

        # prédiction
        yhat_min = float(model.predict(feats)[0])
        hav_km = float(feats.loc[0, "haversine_km"])
        avg_speed = (hav_km / (yhat_min/60)) if yhat_min > 0 else 0.0

        k1,k2,k3 = st.columns(3)
        with k1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.metric("Durée prédite", f"{yhat_min:.1f} min")
            st.markdown("</div>", unsafe_allow_html=True)
        with k2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.metric("Distance (Haversine)", f"{hav_km:.2f} km")
            st.caption("Proxy à vol d'oiseau.")
            st.markdown("</div>", unsafe_allow_html=True)
        with k3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.metric("Vitesse moyenne", f"{avg_speed:.1f} km/h")
            st.markdown("</div>", unsafe_allow_html=True)

        b1,b2,b3,b4 = st.columns(4)
        b1.markdown(f"<span class='badge {'warn' if feats.loc[0,'rush_hour']==1 else 'info'}'>Rush: {int(feats.loc[0,'rush_hour'])}</span>", unsafe_allow_html=True)
        b2.markdown(f"<span class='badge {'ok' if feats.loc[0,'is_weekend']==1 else 'info'}'>Weekend: {int(feats.loc[0,'is_weekend'])}</span>", unsafe_allow_html=True)
        b3.markdown(f"<span class='badge info'>Passengers: {passenger_count}</span>", unsafe_allow_html=True)
        b4.markdown(f"<span class='badge info'>Vendor: {vendor}</span>", unsafe_allow_html=True)

        st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

    else:
        st.info("Saisis deux adresses puis clique **Prédire la durée (minutes)**.")
