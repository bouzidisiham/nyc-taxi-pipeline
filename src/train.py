# src/train.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Colonnes que produit features_duckdb.py
CATEGORICAL = [
    "VendorID", "RatecodeID", "payment_type",
    "is_weekend", "rush_hour",
    "h3_pickup_r8", "h3_dropoff_r8",
    "pickup_hour", "pickup_dow", "pickup_month",
]
NUMERIC = [
    "passenger_count", "trip_distance",
    "haversine_km", "manhattan_km", "bearing_deg",
    "congestion_surcharge", "mta_tax", "tolls_amount",
    "improvement_surcharge", "fare_amount", "extra",
]

def _make_one_hot() -> OneHotEncoder:
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def load_data(path: Path, sample_rows: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    if not path.exists():
        parent = path.parent
        listing = sorted([p.name for p in parent.glob("*.parquet")]) if parent.exists() else []
        raise FileNotFoundError(
            f"Fichier introuvable: {path}\n"
            f"Dossier parent: {parent}\n"
            f"Parquets disponibles ici: {listing}"
        )
    df = pd.read_parquet(path, engine="pyarrow")
    if "trip_duration_min" not in df.columns:
        raise ValueError("La colonne cible 'trip_duration_min' est absente du parquet.")
    # échantillonnage optionnel pour limiter la RAM
    if sample_rows and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=42)
    y = df["trip_duration_min"].to_numpy()
    X = df.drop(columns=["trip_duration_min", "tpep_pickup_datetime", "tpep_dropoff_datetime"], errors="ignore")
    # On garde aussi la colonne temps pour split temporel si présente
    if "tpep_pickup_datetime" in df.columns:
        X = X.join(df[["tpep_pickup_datetime"]])
    return X, y

def build_model(n_estimators: int = 300, max_depth: int | None = None, n_jobs: int = -1) -> Pipeline:
    ohe = _make_one_hot()
    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", ohe),
            ]), CATEGORICAL),
            ("num", SimpleImputer(strategy="median"), NUMERIC),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=n_jobs,
    )
    return Pipeline([("pre", pre), ("model", model)])

def time_split(X: pd.DataFrame, y: np.ndarray, cutoff: pd.Timestamp):
    if "tpep_pickup_datetime" not in X.columns:
        raise ValueError("Split temporel demandé mais 'tpep_pickup_datetime' n'est pas dans les features.")
    ts = pd.to_datetime(X["tpep_pickup_datetime"])
    mask_train = ts < cutoff
    mask_test  = ~mask_train
    if mask_train.sum() == 0 or mask_test.sum() == 0:
        # garde-fou : si cutoff mal choisi, on retombe sur un split aléatoire 80/20
        return train_test_split(X.drop(columns=["tpep_pickup_datetime"]), y, test_size=0.2, random_state=42)
    Xtr = X.loc[mask_train].drop(columns=["tpep_pickup_datetime"])
    Xte = X.loc[mask_test].drop(columns=["tpep_pickup_datetime"])
    ytr = y[mask_train.values]
    yte = y[mask_test.values]
    return Xtr, Xte, ytr, yte

def ensure_expected_columns(feature_names: list[str]):
    miss_num = [c for c in NUMERIC if c not in feature_names]
    miss_cat = [c for c in CATEGORICAL if c not in feature_names]
    if miss_num or miss_cat:
        msg = []
        if miss_num: msg.append(f"Num manquantes: {miss_num}")
        if miss_cat: msg.append(f"Cat manquantes: {miss_cat}")
        raise ValueError("Colonnes attendues absentes du parquet.\n" + "\n".join(msg))

def save_metrics(mae: float, rmse: float, r2: float, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"MAE_min": mae, "RMSE_min": rmse, "R2": r2}, f, indent=2)

def main(
    train_path: Path,
    model_path: Path,
    test_size: float,
    split: str,
    cutoff_year: int | None,
    cutoff_month: int | None,
    cutoff_day: int | None,
    sample_rows: int | None,
    n_estimators: int,
    max_depth: int | None,
    n_jobs: int,
):
    X, y = load_data(train_path, sample_rows=sample_rows)
    ensure_expected_columns(list(X.columns))

    # Split
    if split == "time":
        if not (cutoff_year and cutoff_month and cutoff_day):
            raise ValueError("Pour --split time vous devez fournir --cutoff_year --cutoff_month --cutoff_day.")
        cutoff = pd.Timestamp(year=cutoff_year, month=cutoff_month, day=cutoff_day)
        Xtr, Xte, ytr, yte = time_split(X, y, cutoff)
    else:
        X = X.drop(columns=["tpep_pickup_datetime"], errors="ignore")
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)

    pipe = build_model(n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs)
    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)

    mae = mean_absolute_error(yte, yhat)
    try:
        rmse = mean_squared_error(yte, yhat, squared=False)
    except TypeError:
        rmse = mean_squared_error(yte, yhat) ** 0.5  # compat anciennes versions
    r2 = r2_score(yte, yhat)

    print(f"MAE={mae:.2f} min | RMSE={rmse:.2f} min | R2={r2:.3f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"Saved model to {model_path}")
    save_metrics(mae, rmse, r2, model_path.parent)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--model_path", type=str, default="artifacts/model_rf.pkl")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--sample_rows", type=int, default=None,
                   help="Si défini, échantillonne au plus N lignes pour limiter la RAM.")
    p.add_argument("--split", choices=["random", "time"], default="random")
    p.add_argument("--cutoff_year", type=int, default=None)
    p.add_argument("--cutoff_month", type=int, default=None)
    p.add_argument("--cutoff_day", type=int, default=None)
    # hyperparamètres (optionnels)
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--n_jobs", type=int, default=-1)
    args = p.parse_args()

    main(
        train_path=Path(args.train_path),
        model_path=Path(args.model_path),
        test_size=args.test_size,
        split=args.split,
        cutoff_year=args.cutoff_year,
        cutoff_month=args.cutoff_month,
        cutoff_day=args.cutoff_day,
        sample_rows=args.sample_rows,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=args.n_jobs,
    )
