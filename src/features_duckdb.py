# src/features_duckdb.py
import argparse
from pathlib import Path
import math
import duckdb
import h3

DB_DEFAULT = "data/nyc_taxi.duckdb"

# ---------- Helpers ----------
def _is_none_or_nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))

def _valid_lat(lat) -> bool:
    return not _is_none_or_nan(lat) and -90.0 <= lat <= 90.0

def _valid_lon(lon) -> bool:
    return not _is_none_or_nan(lon) and -180.0 <= lon <= 180.0

def _valid_pair(lat1, lon1, lat2, lon2) -> bool:
    return _valid_lat(lat1) and _valid_lon(lon1) and _valid_lat(lat2) and _valid_lon(lon2)

# ---------- UDF géo (tolérantes aux NULL) ----------
def haversine(lat1, lon1, lat2, lon2) -> float | None:
    if not _valid_pair(lat1, lon1, lat2, lon2):
        return None
    R = 6371.0088
    to_r = math.radians
    lat1, lon1, lat2, lon2 = map(to_r, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return float(2*R*math.asin(math.sqrt(a)))

def manhattan_approx_km(lat1, lon1, lat2, lon2) -> float | None:
    if not _valid_pair(lat1, lon1, lat2, lon2):
        return None
    d1 = haversine(lat1, lon1, lat1, lon2)
    d2 = haversine(lat1, lon1, lat2, lon1)
    return float((d1 or 0.0) + (d2 or 0.0))

def bearing(lat1, lon1, lat2, lon2) -> float | None:
    if not _valid_pair(lat1, lon1, lat2, lon2):
        return None
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    brng = math.degrees(math.atan2(x, y))
    return float((brng + 360) % 360)

def h3_index(lat, lon, res) -> str | None:
    if not (_valid_lat(lat) and _valid_lon(lon)):
        return None
    try:
        return h3.geo_to_h3(lat, lon, int(res))
    except Exception:
        return None

# ---------- SQL helpers ----------
def column_exists(con: duckdb.DuckDBPyConnection, table: str, col: str) -> bool:
    q = """
    SELECT COUNT(*)
    FROM information_schema.columns
    WHERE table_schema='raw' AND table_name=? AND column_name=?
    """
    return con.execute(q, [table, col]).fetchone()[0] > 0

def build_sql(year: int, month: int, h3_res: int, has_latlon_cols: bool) -> str:
    valid_geo_cond = (
        "pickup_latitude IS NOT NULL AND pickup_longitude IS NOT NULL "
        "AND dropoff_latitude IS NOT NULL AND dropoff_longitude IS NOT NULL "
        "AND pickup_latitude BETWEEN -90 AND 90 "
        "AND pickup_longitude BETWEEN -180 AND 180 "
        "AND dropoff_latitude BETWEEN -90 AND 90 "
        "AND dropoff_longitude BETWEEN -180 AND 180"
    )
    has_geo_cols = "TRUE" if has_latlon_cols else "FALSE"

    geo_sql = f"""
    CASE WHEN {has_geo_cols} AND ({valid_geo_cond})
         THEN haversine(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)
         ELSE COALESCE(trip_distance, 0.0) END AS haversine_km,

    CASE WHEN {has_geo_cols} AND ({valid_geo_cond})
         THEN bearing(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)
         ELSE 0.0 END AS bearing_deg,

    CASE WHEN {has_geo_cols} AND ({valid_geo_cond})
         THEN manhattan_approx_km(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)
         ELSE COALESCE(trip_distance, 0.0) END AS manhattan_km,

    CASE WHEN {has_geo_cols} AND ({valid_geo_cond})
         THEN h3_index(pickup_latitude, pickup_longitude, {h3_res})
         ELSE NULL END AS h3_pickup_r8,

    CASE WHEN {has_geo_cols} AND ({valid_geo_cond})
         THEN h3_index(dropoff_latitude, dropoff_longitude, {h3_res})
         ELSE NULL END AS h3_dropoff_r8
    """

    sql = f"""
WITH base AS (
  SELECT
    *,
    GREATEST(1, LEAST(180,
      EXTRACT(EPOCH FROM (tpep_dropoff_datetime - tpep_pickup_datetime)) / 60.0
    ))::DOUBLE AS trip_duration_min,
    EXTRACT(hour  FROM tpep_pickup_datetime)::INT  AS pickup_hour,
    EXTRACT(dow   FROM tpep_pickup_datetime)::INT  AS pickup_dow,
    EXTRACT(month FROM tpep_pickup_datetime)::INT  AS pickup_month
  FROM raw.taxi_trips
  WHERE src_year = {year} AND src_month = {month}
),
geo AS (
  SELECT
    b.*,
    {geo_sql}
  FROM base b
),
final AS (
  SELECT
    trip_duration_min,
    vendor_id AS "VendorID",
    ratecodeid AS "RatecodeID",
    payment_type,
    passenger_count,
    trip_distance,
    haversine_km, manhattan_km, bearing_deg,
    pickup_hour, pickup_dow, pickup_month,
    (pickup_dow >= 5)::INT AS is_weekend,
    (pickup_hour IN (7,8,9,16,17,18))::INT AS rush_hour,
    h3_pickup_r8, h3_dropoff_r8,
    tpep_pickup_datetime, tpep_dropoff_datetime,
    congestion_surcharge, mta_tax, tolls_amount, improvement_surcharge, fare_amount, extra
  FROM geo
)
SELECT * FROM final
"""
    return sql.strip()

def main(db_path: str, out_parquet: Path, year: int, month: int, h3_res: int):
    con = duckdb.connect(db_path)

    # UDFs typées (types Python)
    con.create_function("haversine", haversine, parameters=[float, float, float, float], return_type=float)
    con.create_function("manhattan_approx_km", manhattan_approx_km, parameters=[float, float, float, float], return_type=float)
    con.create_function("bearing", bearing, parameters=[float, float, float, float], return_type=float)
    con.create_function("h3_index", h3_index, parameters=[float, float, int], return_type=str)

    has_latlon_cols = all([
        column_exists(con, "taxi_trips", "pickup_latitude"),
        column_exists(con, "taxi_trips", "pickup_longitude"),
        column_exists(con, "taxi_trips", "dropoff_latitude"),
        column_exists(con, "taxi_trips", "dropoff_longitude"),
    ])

    sql = build_sql(year, month, h3_res, has_latlon_cols)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({sql}) TO '{out_parquet.as_posix()}' (FORMAT PARQUET)")
    print(f"Wrote features to {out_parquet}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default=DB_DEFAULT)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--h3_res", type=int, default=8)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.out) if args.out else Path(f"data/features/train_{args.year}_{args.month:02d}.parquet")
    main(args.db_path, out, args.year, args.month, args.h3_res)
