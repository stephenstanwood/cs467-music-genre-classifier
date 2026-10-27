from functions import *
import wandb
from wandb.integration.keras import WandbMetricsLogger

# BASELINE MODEL WITH 57_FEATURES_3_SEC MODEL
# Reference: Stack Overflow, "python split to 
# train/test/val using GroupShuffleSplit"

# NOTE: Comparing Sam's 57 features extracted from 3-second audio segments
# with GTZAN's 57 features extracted from 3-second audio files.

# ==================== LOAD AND PREPARE DATA ====================

# GTZAN
df_gtzan = pd.read_csv("Data/features_3_sec.csv")

x_gtzan = df_gtzan.drop(columns=["filename", "label", "length"])
y_gtzan = df_gtzan["label"]

# Keep song segments together for group-aware splitting
groups_gtzan = df_gtzan["filename"].str.replace(
    r"\.\d+\.wav$", "", regex=True
)

# PREPROCESSED 57-feature dataset with Dawson's splits
df_pre = pd.read_csv("Data/gtzan_57_features_split.csv")

exclude_columns = [
    "filename", "song_id", "segment", "label", "split"
]

x_pre = df_pre.drop(columns=exclude_columns)
y_pre = df_pre["label"]
groups_pre = df_pre["song_id"]

x_pre = x_pre[x_gtzan.columns]

# ==================== SPLIT GTZAN ====================

# First: separate the test set by song
(
    x_gtzan_train_val,
    x_gtzan_test,
    y_gtzan_train_val,
    y_gtzan_test,
    groups_gtzan_train_val,
    groups_gtzan_test
) = split_by_song(
    x_gtzan, y_gtzan, groups_gtzan,
    test_size=0.2,
    random_state=42
)

# Second: separate validation data from the remaining songs
(
    x_gtzan_train,
    x_gtzan_val,
    y_gtzan_train,
    y_gtzan_val,
    groups_gtzan_train,
    groups_gtzan_val
) = split_by_song(
    x_gtzan_train_val,
    y_gtzan_train_val,
    groups_gtzan_train_val,
    test_size=0.2,
    random_state=43
)

# ==================== SPLIT PREPROCESSED ====================

# NOTE: Use double splits if not using Dawson's .csv splits
""" # First: separate the test set by song
(
    x_pre_train_val,
    x_pre_test,
    y_pre_train_val,
    y_pre_test,
    groups_pre_train_val,
    groups_pre_test
) = split_by_song(
    x_pre, y_pre, groups_pre,
    test_size=0.2,
    random_state=42
)

# Second: separate validation data from the remaining songs
(
    x_pre_train,
    x_pre_val,
    y_pre_train,
    y_pre_val,
    groups_pre_train,
    groups_pre_val
) = split_by_song(
    x_pre_train_val,
    y_pre_train_val,
    groups_pre_train_val,
    test_size=0.2,
    random_state=43
) """

# Using Dawson's split.csv
train_pre = df_pre[df_pre["split"] == "train"]
val_pre = df_pre[df_pre["split"] == "val"]
test_pre = df_pre[df_pre["split"] == "test"]

x_pre_train = train_pre[x_gtzan.columns]
y_pre_train = train_pre["label"]

x_pre_val = val_pre[x_gtzan.columns]
y_pre_val = val_pre["label"]

x_pre_test = test_pre[x_gtzan.columns]
y_pre_test = test_pre["label"]

groups_pre_train = train_pre["song_id"]
groups_pre_val = val_pre["song_id"]
groups_pre_test = test_pre["song_id"]

# ==================== SCALE FEATURES ====================

# GTZAN
(
    x_gtzan_train_sc,
    x_gtzan_val_sc,
    x_gtzan_test_sc,
    gtzan_scaler
) = scale_features(
    x_gtzan_train,
    x_gtzan_val,
    x_gtzan_test
)

# PREPROCESSED
(
    x_pre_train_sc,
    x_pre_val_sc,
    x_pre_test_sc,
    pre_scaler
) = scale_features(
    x_pre_train,
    x_pre_val,
    x_pre_test
)

# ==================== ENCODE LABELS ====================

gtzan_label_encoder = LabelEncoder()
pre_label_encoder = LabelEncoder()

gtzan_label_encoder.fit(y_gtzan)
pre_label_encoder.fit(y_pre)

y_gtzan_train_enc = gtzan_label_encoder.transform(y_gtzan_train)
y_gtzan_val_enc = gtzan_label_encoder.transform(y_gtzan_val)
y_gtzan_test_enc = gtzan_label_encoder.transform(y_gtzan_test)

y_pre_train_enc = pre_label_encoder.transform(y_pre_train)
y_pre_val_enc = pre_label_encoder.transform(y_pre_val)
y_pre_test_enc = pre_label_encoder.transform(y_pre_test)

print("GTZAN genres:", gtzan_label_encoder.classes_)
print("PREPROCESSED genres:", pre_label_encoder.classes_)

# ===================== NEURAL NETWORK ===========================
run = wandb.init(project="music-genre-classifier", job_type="training")

# Model 1: GTZAN Features
model_gtzan = build_model(
    shape=x_gtzan_train_sc.shape[1],
    dense_1=64,
    dense_2=len(gtzan_label_encoder.classes_)
)

history_gtzan = model_gtzan.fit(
    x_gtzan_train_sc,
    y_gtzan_train_enc,
    epochs=30,
    batch_size=32,
    validation_data=(x_gtzan_val_sc, y_gtzan_val_enc),

    callbacks=[WandbMetricsLogger()]
)

# Model 2: PREPROCESSED Features
model_pre = build_model(
    shape=x_pre_train_sc.shape[1],
    dense_1=64,
    dense_2=len(pre_label_encoder.classes_)
)

history_pre = model_pre.fit(
    x_pre_train_sc,
    y_pre_train_enc,
    epochs=30,
    batch_size=32,
    validation_data=(x_pre_val_sc, y_pre_val_enc),

    callbacks=[WandbMetricsLogger()]
)

# ====================== EVALUATE MODEL ===========================
gtzan_results = evaluate_model(
    model_gtzan,
    x_gtzan_test_sc,
    y_gtzan_test_enc,
    gtzan_label_encoder,
    title="GTZAN",
    plot=True
)

pre_results = evaluate_model(
    model_pre,
    x_pre_test_sc,
    y_pre_test_enc,
    pre_label_encoder,
    title="PREPROCESSED",
    plot=True
)

wandb.finish()