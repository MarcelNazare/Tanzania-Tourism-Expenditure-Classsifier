import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.ensemble import HistGradientBoostingClassifier

# ---------------------------------------------------------
# Step 1: Load Datasets
# ---------------------------------------------------------
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
sample_sub = pd.read_csv('SampleSubmission.csv')

target_col = 'cost_category'
id_col = 'Tour_ID'
target_classes = ['High Cost', 'Higher Cost', 'Highest Cost', 'Low Cost', 'Lower Cost', 'Normal Cost']

# ---------------------------------------------------------
# Step 2: Data Cleaning & Feature Engineering Function
# ---------------------------------------------------------
def preprocess_data(df):
    df = df.copy()
    
    # 2a. Fix data entry typos
    df['main_activity'] = df['main_activity'].replace({'Widlife Tourism': 'Wildlife Tourism'})
    
    # 2b. Handle missing values
    df['travel_with'] = df['travel_with'].fillna('Alone')
    df['total_female'] = df['total_female'].fillna(0)
    df['total_male'] = df['total_male'].fillna(0)
    
    # 2c. Feature Engineering
    df['total_people'] = df['total_female'] + df['total_male']
    df['total_nights'] = df['night_mainland'] + df['night_zanzibar']
    
    # Package inclusions sum
    package_cols = [
        'package_transport_int', 'package_accomodation', 'package_food',
        'package_transport_tz', 'package_sightseeing', 'package_guided_tour',
        'package_insurance'
    ]
    df['package_count'] = (df[package_cols] == 'Yes').sum(axis=1)
    
    return df

# Apply processing
train_df = preprocess_data(train)
test_df = preprocess_data(test)

# ---------------------------------------------------------
# Step 3: Categorical Column Preparation
# ---------------------------------------------------------
cat_cols = [
    'country', 'age_group', 'travel_with', 'purpose', 'main_activity', 
    'info_source', 'tour_arrangement', 'package_transport_int', 
    'package_accomodation', 'package_food', 'package_transport_tz', 
    'package_sightseeing', 'package_guided_tour', 'package_insurance', 'first_trip_tz'
]

X = train_df.drop(columns=[id_col, target_col])
y = train_df[target_col]
X_test = test_df.drop(columns=[id_col])

# Convert text/object columns to pandas 'category' type for gradient boosting
for col in cat_cols:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# ---------------------------------------------------------
# Step 4: Stratified Cross-Validation & Modeling
# ---------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((len(train_df), len(target_classes)))
test_preds = np.zeros((len(test_df), len(target_classes)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # Classifier native to categorical feature handling
    model = HistGradientBoostingClassifier(
        categorical_features=cat_cols,
        max_iter=300,
        learning_rate=0.05,
        random_state=42
    )
    
    model.fit(X_tr, y_tr)
    
    # Out-of-fold validation prediction
    oof_preds[val_idx] = model.predict_proba(X_va)
    # Average test set predictions across folds
    test_preds += model.predict_proba(X_test) / skf.n_splits

# Print Out-of-Fold Validation Metric
print(f"OOF Multi-Class Log Loss: {log_loss(y, oof_preds):.4f}")

# ---------------------------------------------------------
# Step 5: Format & Export Submission File
# ---------------------------------------------------------
# Map probability array back to class columns matching sample submission
submission = pd.DataFrame(test_preds, columns=model.classes_)
submission.insert(0, id_col, test_df[id_col])

# Strict column ordering as expected by Zindi
target_order = ['Tour_ID', 'High Cost', 'Higher Cost', 'Highest Cost', 'Low Cost', 'Lower Cost', 'Normal Cost']
submission = submission[target_order]

# Save to CSV
submission.to_csv('final_submission.csv', index=False)
print("Saved final_submission.csv successfully!")