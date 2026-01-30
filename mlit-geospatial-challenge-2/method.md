# 第2回 国土交通省 地理空間情報データチャレンジ

- 不動産の売買価格 `money_room` を回帰で予測する。
- 元データに加えて、国土数値情報の「駅別乗降者数」データを取り込み、立地情報を強化している。
- CatBoost + Optuna でハイパラチューニング。

---

## データ前処理

### 1. 欠損・完全欠損列

- train/test の欠損率・欠損数を集計。
- train/test のどちらか、あるいは両方で **100%欠損の列を削除**。
- 以降は `train_clean` / `test_clean` を使用。

### 2. 外部データ（駅別乗降者数）の追加

- 国土数値情報の駅データ（2019–2023年）を読み込み。
  - 年ごとに乗降客数カラム名が違うので、共通で `乗降客数` 列にリネーム。
  - 同一駅（駅名 + geometry）が複数行ある場合は、乗降客数が最大の行だけ残す。
- 各物件について:
  - `target_ym` から年（2019〜2023）を取得し、その年の駅データと紐付ける。
  - `lon`/`lat` からポイントの GeoDataFrame を作成し、日本用メートル座標系（EPSG:6677）に変換。
  - 各物件を中心に **800m の円** を作り、駅ポイントと空間結合。
  - 800m 圏内にある駅について
    - 駅の個数 `station_count_800m`
    - 乗降客数合計 `total_passengers_800m`
    を集計して特徴量として付与。
- `train_clean` / `test_clean` の両方に上記2特徴量を追加。

### 3. 特徴量の分類と変換

- 手動で以下のグループに分ける:
  - `numerical_columns`: 面積・距離・金額などの連続値。
  - `cat_num_cols`: 整数だがカテゴリとして扱うコード（用途地域、構造、築種別など）。
  - `slash_cols`: `"1/3/5"` のようなスラッシュ区切りのタグ列。
  - `ym_cols`: yyyymm 形式の年月列（築年月、リフォーム年月など）。
  - `datetime_cols`: 日時列（データ作成日・公開日など）。
  - `text_cols_to_drop`: 名前・住所・備考など今回は使わないテキスト列。
  - `id_cols_to_drop`: building_id / unit_id / bukken_id などの ID 列。
- train/test を縦結合して `all_data` を作り、以下をまとめて実行：

#### 日付系の展開

- `target_ym` → `target_date`, `target_year`, `target_month`。
- `ym_cols`:
  - yyyymm を datetime に変換し、
    - `*_year`, `*_month`,
    - `*_since_target_days`（対象年月からの経過日数）
    を作る。
  - `year_built` から築年数 `building_age_years` も計算。
- `datetime_cols`:
  - datetime に変換し、
    - `*_year`, `*_month`, `*_day`, `*_dow`（曜日）
    - `*_since_target_days`
    を作る。
- 変換に使った元の年月・日時列は削除。

#### スラッシュ区切り列の展開

- `slash_cols` の各列について:
  - `/` で分割してタグのリスト化。
  - `MultiLabelBinarizer` で 0/1 のマルチホットに展開。
    - 列名は `元列名_tag_タグID`。
  - タグ数 `元列名_tag_count` も 1 列追加。
  - 元の文字列列は削除。

#### 型整備・カテゴリ化

- `text_cols_to_drop` と `id_cols_to_drop` を削除。
- `numerical_columns` は `to_numeric` で数値に揃える。
- それ以外の int/float 型の列も数値特徴量 `numeric_all` に含める。
- `cat_num_cols` は `category` 型にキャスト。
- 残りの object 型もすべて `category` にキャストし、`categorical_features` としてまとめる。
- `all_data` を
  - 先頭 `len(train_clean)` 行 → `X`
  - 残り → `X_test`
  に分割。
- CatBoost 用に `categorical_features` の NaN を `"__MISSING__"` に置き換え、列位置（インデックス）を `cat_feature_indices` として保持。

---

## モデリング

### 1. 目的変数

- 予測対象: `money_room`
- 元スケール: `y_raw`
- 学習スケール: `y_log = log1p(y_raw)`
- 予測後は `expm1` で元スケールに戻す。

### 2. CatBoost + Optuna でハイパラ探索

- GPU / CPU はフラグ `USE_GPU` で切り替え。
- Optuna の `objective` 関数では、以下をサンプリング:
  - `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`
  - `bagging_temperature`, `random_strength`
  - `loss_function = "RMSE"`, `eval_metric = "RMSE"`
- `KFold(n_splits=5, shuffle=True, random_state=42)` で5-fold CV。
- 各foldで:
  - `y_log` で学習。
  - 予測値を `expm1` で戻し、`y_raw` と比較して MAPE を計算。
- 5fold の MAPE の平均を Optuna の最適化対象にして、30 trial 実行。
- best MAPE と best_params を出力。

### 3. ベストパラメータでの CV + テスト予測

- `study.best_params` をベースに CatBoost パラメータを確定。
- 同じく 5-fold CV を回して:
  - 各 fold の検証 MAPE を表示。
  - 各 fold で `X_test` を予測し、`test_pred_list` に保存。
- fold 予測を平均して `test_pred_mean` を作成。

---

## 特徴量重要度と提出ファイル

- 全学習データ `X, y_log` で最終モデル `model_full` を学習。
- `get_feature_importance()` で FI を取り出し、`feature`/`importance` の DataFrame にする。
- 重要度順にソートし、上位40件をバーでプロット。
- `test_pred_mean` と `test["id"]` から提出用 DataFrame を作成。
  - `id` を6桁ゼロ埋め。
  - タイムスタンプ付きのファイル名で CSV を出力（ヘッダ無し）。

---
