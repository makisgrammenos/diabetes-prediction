import streamlit as st
import numpy as np
import pandas as pd
from ml import run_classification, run_clustering
def load_data(file):
    data = pd.read_csv(file)
    return data


def main():
    st.title("Diabetes classification")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    uploaded_file = st.file_uploader("Choose a file", type=['csv'])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write(df.head())
        target_col = st.selectbox("Select the target column", df.columns)
        
        # dataset description
        st.subheader("Dataset Description")
        st.write('Number of rows: ', df.shape[0])
        st.write('Number of columns: ', df.shape[1])
        
        
        st.subheader("Classification")
        st.write(f"Target column: {target_col}")
        
        
        
        st.write("Default parameters are the following: n_estimators=500, max_depth=10")
        n_estimators = st.slider("Select the number of estimators for random forest", min_value=10, max_value=1000, step=10, value=500)
        max_depth = st.slider("Select the maximum depth for random forest", min_value=1, max_value=50, value=10)
        cross_validation = st.checkbox(" 5-fold cross-validation", value=False)
        
        if not cross_validation:
            train_size = st.slider("Select the training set size", min_value=0.1, max_value=0.9, step=0.1, value=0.8)
            test_size = 1 - train_size
            st.write(f"Training set size: {train_size}")
            st.write(f"Test set size: {np.round(test_size, 2)}")
        else:
            train_size = None
            test_size = None
        
            
            
        if st.button("Run Random Forest classification"):
            cv_results = run_classification(df, target_col, n_estimators, max_depth, cross_validation, train_size, test_size)
            st.write("Random Forest Cross-Validation Results:")
            # for i, result in enumerate(cv_results):
            #     st.write(f"Fold {i+1}:")
            #     st.write(f"Precision: {result[0]}")
            #     st.write(f"Recall: {result[1]}")
            st.dataframe(cv_results)
        st.subheader("Clustering")
        n_clusters = st.slider("Select the number of clusters for k-means", min_value=2, max_value=10, value=2)
        if st.button("Run Clustering"):
            st.write(f"Number of clusters: {n_clusters}")
            centers, silhouette_avg, calinski_harabasz = run_clustering(df, n_clusters,target_col)
            st.write(f"K-Means Clustering Results:")
            st.write(f"Silhouette Score: {silhouette_avg}")
            st.write(f"Calinski-Harabasz Index: {calinski_harabasz}")
            st.write(f"Cluster Centers:")
            st.write(centers)
if __name__ == '__main__':
    main()