##### univariate statistical test, variables are ranked ######
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
from sklearn.feature_selection import SelectKBest
##### recursive feature elimination #####
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
###### machine learning-based method based on feature importance
from sklearn.ensemble import RandomForestClassifier


def feature_evaluation_show_all(X, y, measure):
    if measure == 'f_test':
        f_scores, p_values  =  f_classif(X, y)
        feature_f_scores = dict(zip(X.columns, f_scores))
        sorted_features = sorted(feature_f_scores.items(), key=lambda x: x[1], reverse=True)
        for feature, f_score in sorted_features:
            print(f"{feature}: {f_score}")
    elif measure == 'chi2':
        chi2_scores, p_values =  chi2(X, y)
        feature_chi2_scores = dict(zip(X.columns, chi2_scores))
        sorted_features = sorted(feature_chi2_scores.items(), key=lambda x: x[1], reverse=True)
        for feature, chi2_score in sorted_features:
            print(f"{feature}: {chi2_score}")
    elif measure == 'mutual info':
        info_gains =  mutual_info_classif(X, y)
        feature_info_gains = dict(zip(X.columns, info_gains))
        sorted_features = sorted(feature_info_gains.items(), key=lambda x: x[1], reverse=True)
        for feature, info_gain in sorted_features:
            print(f"{feature}: {info_gain}")
    elif measure == 'rfe-rf':
        rf_classifier = RandomForestClassifier(random_state=42)
        rfe = RFE(estimator=rf_classifier, n_features_to_select=1)
        rfe.fit(X, y)
        feature_ranking = list(zip(rfe.ranking_, X.columns))
        feature_ranking.sort(key=lambda x: x[0])
        for rank, feature in feature_ranking:
            print(f"Rank {rank}: {feature}")
    elif measure == 'rfe-linear':
        l_classifier = LinearRegression()
        rfe = RFE(estimator=l_classifier, n_features_to_select=1)
        rfe.fit(X, y)
        feature_ranking = list(zip(rfe.ranking_, X.columns))
        feature_ranking.sort(key=lambda x: x[0])
        for rank, feature in feature_ranking:
            print(f"Rank {rank}: {feature}")
    elif measure == 'feature_importance':
        rf_classifier = RandomForestClassifier(random_state=42)
        rf_classifier.fit(X, y)
        importances = rf_classifier.feature_importances_
        feature_importance = list(zip(X.columns, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        for feature, importance in feature_importance:
            print(f"{feature}: {importance}")



def feature_evaluation_select_k(X, y, measure, k):

    if measure == 'f_test':
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X, y)
        feature_indices = selector.get_support(indices=True)
        selected_features = X.columns[feature_indices]
        X_new = X[selected_features]
        return selected_features, X_new
    elif measure == 'chi2':
        selector = SelectKBest(chi2, k=k)
        selector.fit(X, y)
        feature_indices = selector.get_support(indices=True)
        selected_features = X.columns[feature_indices]
        X_new = X[selected_features]
        return selected_features, X_new
    elif measure == 'mutual info':
        selector = SelectKBest(mutual_info_classif, k=k)
        selector.fit(X, y)
        feature_indices = selector.get_support(indices=True)
        selected_features = X.columns[feature_indices]
        X_new = X[selected_features]
        return selected_features, X_new
