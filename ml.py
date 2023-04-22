import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, silhouette_score, calinski_harabasz_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import time


def run_classification(df, target_col, n_estimators, max_depth, cross_validation, train_size=None, test_size=None):
    # Split the data into features and target
    X = df.drop([target_col], axis=1)
    y = df[target_col]

    # Train a random forest classifier
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    with st.spinner('Training model...'):
        if cross_validation:
            for i, (train_index, test_index) in enumerate(skf.split(X, y)):
                # time.sleep(1)
                # progress = (i + 1) / 5
                # st.progress(progress)
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                precision = precision_score(y_test, y_pred, average='macro')
                recall = recall_score(y_test, y_pred, average='macro')
                cv_results.append((precision, recall))
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            cv_results.append((precision, recall))
    st.success('Training completed!')
    cv_results = pd.DataFrame(cv_results, columns=['Precision', 'Recall'])
    return cv_results

def run_clustering(df, n_clusters,target_col):
    # Cluster the data using k-means
    kmeans = KMeans(n_clusters=n_clusters)
    df = df.drop([target_col], axis=1)
    with st.spinner('Clustering data...'):
        # time.sleep(1)
        kmeans.fit(df)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Calculate two clustering metrics
        silhouette_avg = silhouette_score(df, labels)
        calinski_harabasz = calinski_harabasz_score(df, labels)

    return centers, silhouette_avg, calinski_harabasz

