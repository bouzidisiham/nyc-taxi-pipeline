import argparse
from pathlib import Path
import duckdb
from textwrap import dedent
from src.utils_io import get_logger


DB_DEFAULT = "data/nyc_taxi.duckdb"
LOG = get_logger()

# -- 1) DDL : schémas + table cible + vue catalogue ---------------------------
DDL_SETUP = dedent("""
PRAGMA enable_progress_bar=false;

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS mart;

CREATE TABLE IF NOT EXISTS raw.taxi_trips (
    vendor_id             INTEGER,
    tpep_pickup_datetime  TIMESTAMP,
    tpep_dropoff_datetime TIMESTAMP,
    passenger_count       INTEGER,
    trip_distance         DOUBLE,
    ratecodeid            INTEGER,
    pulocationid          INTEGER,
    dolocationid          INTEGER,
    payment_type          INTEGER,
    fare_amount           DOUBLE,
    extra                 DOUBLE,
    mta_tax               DOUBLE,
    tip_amount            DOUBLE,
    tolls_amount          DOUBLE,
    improvement_surcharge DOUBLE,
    total_amount          DOUBLE,
    congestion_surcharge  DOUBLE,
    store_and_fwd_flag    VARCHAR,
    pickup_latitude       DOUBLE,
    pickup_longitude      DOUBLE,
    dropoff_latitude      DOUBLE,
    dropoff_longitude     DOUBLE,
    src_year              INTEGER,
    src_month             INTEGER,
    src_file              VARCHAR,
    _hash_key             BLOB
);

CREATE VIEW IF NOT EXISTS mart.vw_taxi_trips_schema AS
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema='raw' AND table_name='taxi_trips'
ORDER BY ordinal_position;
""")

# -- 2) Vues "mart" pour analyser rapidement ----------------------------------
VIEWS_SQL = dedent("""
CREATE OR REPLACE VIEW mart.vw_daily_stats AS
SELECT DATE_TRUNC('day', tpep_pickup_datetime) AS day,
       COUNT(*) AS trips,
       AVG(EXTRACT(EPOCH FROM (tpep_dropoff_datetime - tpep_pickup_datetime))/60.0) AS avg_duration_min,
       AVG(trip_distance) AS avg_dist_km,
       AVG(total_amount) AS avg_total
FROM raw.taxi_trips
GROUP BY 1 ORDER BY 1;

CREATE OR REPLACE VIEW mart.vw_top_zones AS
SELECT pulocationid, COUNT(*) AS trips, AVG(total_amount) AS avg_total
FROM raw.taxi_trips
GROUP BY 1 ORDER BY trips DESC LIMIT 50;
""")

# -- 3) Lister les colonnes présentes dans le Parquet -------------------------
def columns_in_parquet(con: duckdb.DuckDBPyConnection, parquet_path: Path):
    return (
        con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{parquet_path.as_posix()}')"
        )
        .fetchdf()["column_name"]
        .tolist()
    )

# -- 4) Vue source mensuelle normalisée ---------------------------------------
def register_month_view(con: duckdb.DuckDBPyConnection, parquet_path: Path, year: int, month: int):
    cols = set(columns_in_parquet(con, parquet_path))
    def exists(c: str) -> bool: return c in cols

    sql = f"""
    CREATE OR REPLACE VIEW v_src AS
    SELECT
      CAST(VendorID AS INTEGER)                AS vendor_id,
      CAST(tpep_pickup_datetime AS TIMESTAMP)  AS tpep_pickup_datetime,
      CAST(tpep_dropoff_datetime AS TIMESTAMP) AS tpep_dropoff_datetime,
      CAST(passenger_count AS INTEGER)         AS passenger_count,
      CAST(trip_distance AS DOUBLE)            AS trip_distance,
      CAST(RatecodeID AS INTEGER)              AS ratecodeid,
      CAST(PULocationID AS INTEGER)            AS pulocationid,
      CAST(DOLocationID AS INTEGER)            AS dolocationid,
      CAST(payment_type AS INTEGER)            AS payment_type,
      CAST(fare_amount AS DOUBLE)              AS fare_amount,
      CAST(extra AS DOUBLE)                    AS extra,
      CAST(mta_tax AS DOUBLE)                  AS mta_tax,
      CAST(tip_amount AS DOUBLE)               AS tip_amount,
      CAST(tolls_amount AS DOUBLE)             AS tolls_amount,
      CAST(improvement_surcharge AS DOUBLE)    AS improvement_surcharge,
      CAST(total_amount AS DOUBLE)             AS total_amount,
      CAST(congestion_surcharge AS DOUBLE)     AS congestion_surcharge,
      CAST(store_and_fwd_flag AS VARCHAR)      AS store_and_fwd_flag,
      {year}  AS src_year,
      {month} AS src_month,
      '{parquet_path.name}' AS src_file,
      {('CAST(pickup_latitude AS DOUBLE)' if exists('pickup_latitude') else 'NULL')}   AS pickup_latitude,
      {('CAST(pickup_longitude AS DOUBLE)' if exists('pickup_longitude') else 'NULL')} AS pickup_longitude,
      {('CAST(dropoff_latitude AS DOUBLE)' if exists('dropoff_latitude') else 'NULL')} AS dropoff_latitude,
      {('CAST(dropoff_longitude AS DOUBLE)' if exists('dropoff_longitude') else 'NULL')} AS dropoff_longitude
    FROM read_parquet('{parquet_path.as_posix()}');
    """
    con.execute(sql)

# -- 5) UPSERT du mois (dédoublonnage via hash) --------------------------------
def upsert_month(con, year: int, month: int):
    con.execute("""
        CREATE OR REPLACE TABLE tmp_stage AS
        SELECT
          *,
          CAST(sha256(concat_ws('|',
            COALESCE(CAST(vendor_id AS VARCHAR), ''),
            COALESCE(CAST(tpep_pickup_datetime AS VARCHAR), ''),
            COALESCE(CAST(tpep_dropoff_datetime AS VARCHAR), ''),
            COALESCE(CAST(passenger_count AS VARCHAR), ''),
            COALESCE(CAST(trip_distance AS VARCHAR), ''),
            COALESCE(CAST(total_amount AS VARCHAR), '')
          )) AS BLOB) AS _hash_key
        FROM v_src;
    """)

    # supprime doublons du même mois (mêmes hash)
    con.execute("""
        DELETE FROM raw.taxi_trips
        WHERE src_year = ? AND src_month = ?
          AND _hash_key IN (SELECT _hash_key FROM tmp_stage);
    """, [year, month])

    # insertion avec mapping explicite des colonnes
    con.execute("""
        INSERT INTO raw.taxi_trips (
            vendor_id, tpep_pickup_datetime, tpep_dropoff_datetime, passenger_count,
            trip_distance, ratecodeid, pulocationid, dolocationid, payment_type,
            fare_amount, extra, mta_tax, tip_amount, tolls_amount,
            improvement_surcharge, total_amount, congestion_surcharge,
            store_and_fwd_flag, pickup_latitude, pickup_longitude,
            dropoff_latitude, dropoff_longitude, src_year, src_month, src_file, _hash_key
        )
        SELECT
            vendor_id, tpep_pickup_datetime, tpep_dropoff_datetime, passenger_count,
            trip_distance, ratecodeid, pulocationid, dolocationid, payment_type,
            fare_amount, extra, mta_tax, tip_amount, tolls_amount,
            improvement_surcharge, total_amount, congestion_surcharge,
            store_and_fwd_flag, pickup_latitude, pickup_longitude,
            dropoff_latitude, dropoff_longitude, src_year, src_month, src_file, _hash_key
        FROM tmp_stage;
    """)

    con.execute("DROP TABLE IF EXISTS tmp_stage;")

# -- 6) Petits checks qualité ---------------------------------------------------
def run_quality(con):
    print("=== DATA QUALITY ===")

    # 1) Volume + bornes de dates
    print(con.execute("""
        SELECT
            COUNT(*) AS n_rows,
            MIN(tpep_pickup_datetime) AS min_pickup,
            MAX(tpep_pickup_datetime) AS max_pickup
        FROM raw.taxi_trips;
    """).fetchdf())

    print(con.execute("""
        SELECT 'pickup_ts' AS col,
               CASE WHEN COUNT(*) = 0 THEN 0.0
                    ELSE 100.0 * SUM(CASE WHEN tpep_pickup_datetime  IS NULL THEN 1 ELSE 0 END) / COUNT(*)
               END AS null_pct
        FROM raw.taxi_trips
        UNION ALL
        SELECT 'dropoff_ts',
               CASE WHEN COUNT(*) = 0 THEN 0.0
                    ELSE 100.0 * SUM(CASE WHEN tpep_dropoff_datetime IS NULL THEN 1 ELSE 0 END) / COUNT(*)
               END
        FROM raw.taxi_trips
        UNION ALL
        SELECT 'trip_distance',
               CASE WHEN COUNT(*) = 0 THEN 0.0
                    ELSE 100.0 * SUM(CASE WHEN trip_distance        IS NULL THEN 1 ELSE 0 END) / COUNT(*)
               END
        FROM raw.taxi_trips;
    """).fetchdf())

    # 3) Répartition par mois
    print(con.execute("""
        SELECT src_year, src_month, COUNT(*) AS n
        FROM raw.taxi_trips
        GROUP BY 1,2
        ORDER BY 1,2;
    """).fetchdf())



# -- 7) Main --------------------------------------------------------------------
def main(db_path: str, raw_dir: str, year: int, month: int):
    LOG.info("YO")
    # S'assurer que le dossier de la DB existe (utile en local)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db_path)
    con.execute(DDL_SETUP)

    parquet = Path(raw_dir) / f"yellow_tripdata_{year}-{month:02d}.parquet"
    LOG.info(f"PATH PARQUET {parquet}")
    if not parquet.exists():
        raise FileNotFoundError(f"Missing parquet: {parquet}")

    register_month_view(con, parquet, year, month)
    upsert_month(con, year, month)

    con.execute("ANALYZE raw.taxi_trips;")
    con.execute(VIEWS_SQL)

    run_quality(con)
    print("Ingestion OK → raw.taxi_trips (DuckDB)")

# -- 8) Entrée CLI --------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default=DB_DEFAULT)
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--month", type=int, default=1)
    args = parser.parse_args()

    Path(args.raw_dir).mkdir(parents=True, exist_ok=True)
    main(args.db_path, args.raw_dir, args.year, args.month)
