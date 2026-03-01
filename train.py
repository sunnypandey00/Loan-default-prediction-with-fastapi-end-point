import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

def train_model():
    print("Loading data...")
    try:
        df = pd.read_csv('Loan_Default.csv')
    except FileNotFoundError:
        print("dataset.csv not found in this folder.")
        return

    selected_features = ['Credit_Score', 'LTV', 'dtir1', 'loan_type', 'age', 'Region']
    X = df[selected_features]
    y = df['Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_features = ['Credit_Score', 'LTV', 'dtir1']
    categorical_features = ['loan_type', 'age', 'Region']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=169,
            learning_rate=0.12495365391494376,
            max_depth=4,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    joblib.dump(model, 'best_model.pkl')
    print("Model saved as best_model.pkl")
    
    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_model()