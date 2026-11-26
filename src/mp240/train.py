import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .clean import clean_train
from .config import ROOT, load_params, train_sha256

def main():
    p = load_params()
    d = p["data"]
    f = p["features"]

    raw = pd.read_csv(ROOT / d["train"])
    df = clean_train(raw, p)

    X = df[f["numeric"] + f["categorical"]]
    y = df[d["target"]]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric, f["numeric"]),
        ("cat", categorical, f["categorical"])
    ])

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=200, max_depth=12, random_state=42
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3,
            random_state=42
        )
    }

    tracking_db = (ROOT / "mlflow.db").as_posix()
    mlflow.set_tracking_uri("sqlite:///" + tracking_db)
    mlflow.set_experiment("MP240_Cab_Pickup_Time")

    results = []
    best_mae = float("inf")
    best_model = None
    best_run_id = None
    model_name = "MP240_Cab_Pickup_Model"

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        with mlflow.start_run(run_name=name) as run:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_val)

            mae = mean_absolute_error(y_val, predictions)
            rmse = mean_squared_error(y_val, predictions) ** 0.5
            r2 = r2_score(y_val, predictions)

            mlflow.log_params({
                "model": name,
                "random_state": 42,
                "validation_size": 0.2
            })
            mlflow.log_metrics({
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2)
            })
            mlflow.log_param("data_sha256", train_sha256())

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format="cloudpickle"
            )

            results.append({
                "model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

            print(name, "MAE:", round(mae, 4),
                  "RMSE:", round(rmse, 4), "R2:", round(r2, 4))

            if mae < best_mae:
                best_mae = mae
                best_model = pipeline
                best_run_id = run.info.run_id

    model_path = ROOT / "models" / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)

    registered = mlflow.register_model(
        f"runs:/{best_run_id}/model", model_name
    )

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        model_name, "champion", registered.version
    )

    results.sort(key=lambda x: x["MAE"])

    report = {
        "best_model": results[0]["model"],
        "best_MAE": results[0]["MAE"],
        "best_RMSE": results[0]["RMSE"],
        "best_R2": results[0]["R2"],
        "models": results,
        "registered_model": model_name,
        "champion_version": registered.version
    }

    (ROOT / "reports" / "training_results.json").write_text(
        json.dumps(report, indent=2)
    )

    print("\nTRAINING COMPLETE")
    print(json.dumps(report, indent=2))
    print("Saved:", model_path)

if __name__ == "__main__":
    main()


