import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier, Pool

# ---------------------------------------------------------
# 1. Load Datasets & Extract Exact Output Specification
# ---------------------------------------------------------
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
sample_sub = pd.read_csv('SampleSubmission.csv')

# Extract ID column and target classes directly from Sample Submission
id_col = sample_sub.columns[0]  # 'Tour_ID'
target_classes = list(sample_sub.columns[1:])  # ['High Cost', 'Higher Cost', 'Highest Cost', 'Low Cost', 'Lower Cost', 'Normal Cost']

# Create mapping from target class string to class index
class_to_idx = {cls_name: i for i, cls_name in enumerate(target_classes)}
idx_to_class = {i: cls_name for i, cls_name in enumerate(target_classes)}

# Map train target column to integer indices matching sample submission
train['target'] = train['cost_category'].map(class_to_idx)

# ---------------------------------------------------------
# 2. Data Preprocessing & Feature Engineering
# ---------------------------------------------------------
def preprocess_data(df):
    df = df.copy()
    
    # Fix dataset typos
    if 'main_activity' in df.columns:
        df['main_activity'] = df['main_activity'].replace({'Widlife Tourism': 'Wildlife Tourism'})
    
    # Handle missing values explicitly
    df['travel_with'] = df['travel_with'].fillna('Alone')
    df['total_female'] = df['total_female'].fillna(0)
    df['total_male'] = df['total_male'].fillna(0)
    
    # Domain Feature Engineering
    df['total_people'] = df['total_female'] + df['total_male']
    df['total_nights'] = df['night_mainland'] + df['night_zanzibar']
    
    # Sum of package perks included
    package_cols = [
        'package_transport_int', 'package_accomodation', 'package_food',
        'package_transport_tz', 'package_sightseeing', 'package_guided_tour',
        'package_insurance'
    ]
    df['package_count'] = (df[package_cols] == 'Yes').sum(axis=1)
    
    return df

train_df = preprocess_data(train)
test_df = preprocess_data(test)

# ---------------------------------------------------------
# 3. Categorical Feature Specification
# ---------------------------------------------------------
cat_features = [
    'country', 'age_group', 'travel_with', 'purpose', 'main_activity', 
    'info_source', 'tour_arrangement', 'package_transport_int', 
    'package_accomodation', 'package_food', 'package_transport_tz', 
    'package_sightseeing', 'package_guided_tour', 'package_insurance', 'first_trip_tz'
]

# Ensure categorical columns are strings for CatBoost
for col in cat_features:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

features = [col for col in train_df.columns if col not in [id_col, 'cost_category', 'target']]

X = train_df[features]
y = train_df['target']
X_test = test_df[features]

# ---------------------------------------------------------
# 4. Stratified K-Fold Cross-Validation & Training
# ---------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((len(train_df), len(target_classes)))
test_preds = np.zeros((len(test_df), len(target_classes)))

test_pool = Pool(X_test, cat_features=cat_features)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Training Fold {fold + 1} ---")
    
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    val_pool = Pool(X_va, y_va, cat_features=cat_features)
    
    model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.04,
        depth=6,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        random_seed=42,
        task_type='CPU',  # Set to 'GPU' if running on a GPU instance
        verbose=200
    )
    
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=100,
        use_best_model=True
    )
    
    oof_preds[val_idx] = model.predict_proba(val_pool)
    test_preds += model.predict_proba(test_pool) / skf.n_splits

# Print Out-of-Fold validation score
cv_loss = log_loss(y, oof_preds)
print(f"\n==========================================")
print(f"Overall OOF Log Loss: {cv_loss:.5f}")
print(f"==========================================")

# ---------------------------------------------------------
# 5. Export Exactly Aligned Submission
# ---------------------------------------------------------
# Create prediction DataFrame with columns ordered by class index
submission = pd.DataFrame(test_preds, columns=[idx_to_class[i] for i in range(len(target_classes))])
submission.insert(0, id_col, test_df[id_col])

# Strictly match SampleSubmission.csv column order
submission = submission[sample_sub.columns]

# Ensure row order matches SampleSubmission.csv
submission = sample_sub[[id_col]].merge(submission, on=id_col, how='left')

# Verification checks
assert list(submission.columns) == list(sample_sub.columns), "Column alignment mismatch!"
assert (submission[id_col] == sample_sub[id_col]).all(), "ID order mismatch!"
assert submission.isnull().sum().sum() == 0, "Submission contains null values!"

# Save final CSV file
submission.to_csv('catboost_submission.csv', index=False)
print("catboost_submission.csv saved successfully!")