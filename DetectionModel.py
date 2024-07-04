import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from scipy.stats import randint


data = pd.read_csv('creditcard.csv')
X = data.drop(['Class'], axis=1)
y = data['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train_scaled, y_train)

param_dist = {
    'n_estimators': randint(50, 200),
    'max_features': ['auto', 'sqrt'],
    'max_depth': randint(4, 10),
    'criterion': ['gini', 'entropy']
}

random_search = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),param_distributions=param_dist,scoring='roc_auc', n_iter=20,  cv=3, n_jobs=-1,random_state=42)

random_search.fit(X_resampled, y_resampled)


best_params = random_search.best_params_
best_model = random_search.best_estimator_


y_pred_best = best_model.predict(X_test_scaled)
y_pred_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred_best))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_prob_best))


