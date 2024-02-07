import pandas as pd
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


# Ignore the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


def calculate_clustering_scores(df_selected, kmeans):
    clusters = kmeans.fit_predict(df_selected)

    silhouette_avg = silhouette_score(df_selected, clusters)
    
    intra_distances = kmeans.transform(df_selected)
    intra_cluster_distance = np.mean(np.min(intra_distances, axis=1))
    
    cluster_centers = kmeans.cluster_centers_
    inter_cluster_distances = cdist(cluster_centers, cluster_centers, 'euclidean')
    inter_cluster_distance = np.min(inter_cluster_distances[np.nonzero(inter_cluster_distances)])
    
    return silhouette_avg, intra_cluster_distance, inter_cluster_distance


def plot_cluster_metrics(df):
    sse = []
    silhouette_scores = []
    intra_cluster_distances = []
    inter_cluster_distances = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
        silhouette_avg, intra_cluster_distance, inter_cluster_distance = calculate_clustering_scores(df, kmeans)
        silhouette_scores.append(silhouette_avg)
        intra_cluster_distances.append(intra_cluster_distance)
        inter_cluster_distances.append(inter_cluster_distance)


    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    axs[0, 0].plot(range(2, 11), sse, 'bx-')
    axs[0, 0].set_xlabel('Nombre de clusters')
    axs[0, 0].set_ylabel('SSE (Somme des carrés des erreurs)')
    axs[0, 0].set_title('La Méthode du Coude pour déterminer le nombre optimal de clusters')

    axs[0, 1].plot(range(2, 11), silhouette_scores, 'bx-')
    axs[0, 1].set_xlabel('Nombre de clusters')
    axs[0, 1].set_ylabel('Score de silhouette')
    axs[0, 1].set_title('Score de silhouette par rapport au nombre de clusters, à maximiser')

    axs[1, 0].plot(range(2, 11), intra_cluster_distances, 'bx-')
    axs[1, 0].set_xlabel('Nombre de clusters')
    axs[1, 0].set_ylabel('Distance intra-cluster')
    axs[1, 0].set_title('Distance intra-cluster par rapport au nombre de clusters, à minimiser')

    axs[1, 1].plot(range(2, 11), inter_cluster_distances, 'bx-')
    axs[1, 1].set_xlabel('Nombre de clusters')
    axs[1, 1].set_ylabel('Distance inter-cluster')
    axs[1, 1].set_title('Distance inter-cluster par rapport au nombre de clusters, à maximiser')

    plt.tight_layout()

    plt.show()



def make_clusters(df, n):
    data = df.iloc[:, 3:] 
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(data_standardized)

    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data_standardized)

    fig = px.scatter_3d(
        df,
        x=data_pca[:, 0],
        y=data_pca[:, 1],
        z=data_pca[:, 2],
        color='cluster',
        labels={'color': 'Cluster'},
        title='Clustering of Respondents (Interactive 3D)',
        width=1200,
        height=1200,
        size_max=35,
        opacity=1,
        size="weight"
    )

    fig.show()

    return kmeans

def feature_importance(df):
    data = df.iloc[:, 3:-1] 
    
    data = data.apply(pd.to_numeric, errors='ignore')
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)

    model = RandomForestClassifier(random_state=42)
    model.fit(data_standardized, df['cluster'])

    feature_importances = model.feature_importances_

    plt.figure(figsize=(12, 6))
    sns.barplot(x=data.columns, y=feature_importances, palette='viridis')
    plt.title('Feature Importances across Clusters')
    plt.xticks(rotation=45, ha='right')
    plt.show()

    return feature_importances


def features_per_cluster(df, kmeans, n):
    cluster_centers = kmeans.cluster_centers_
    
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=df.columns)
    
    most_important_features = {}
    
    for cluster in range(n):
        sorted_features = cluster_centers_df.iloc[cluster].abs().sort_values(ascending=False)
        
        top_features = sorted_features.head(5)  # Change 5 to the desired number of top features
        
        most_important_features[f'Cluster {cluster}'] = top_features.index.tolist()

    print("Most Important Features by Cluster:")
    for cluster, features in most_important_features.items():
        print(f"{cluster}: {features}")


def compare_cluster_describe(df, cluster_column, column_to_compare, statistic='mean'):
    clusters = df[cluster_column].unique()

    cluster_stats = pd.DataFrame(index=clusters, columns=['Cluster', 'Value'])

    for cluster in clusters:
        cluster_data = df[df[cluster_column] == cluster]
        value = cluster_data[column_to_compare].describe().loc[statistic]
        cluster_stats.loc[cluster] = [cluster, value]

    plt.figure(figsize=(10, 6))
    plt.bar(cluster_stats['Cluster'], cluster_stats['Value'], color='skyblue')
    plt.title(f'{statistic.capitalize()} of {column_to_compare} by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(f'{statistic.capitalize()} of {column_to_compare}')
    plt.show()

def compare_cluster_answers(df, cluster_column, column_to_compare):
    clusters = sorted(df[cluster_column].unique())

    cluster_counts = pd.DataFrame(index=clusters, columns=['Cluster'] + df[column_to_compare].unique().tolist())

    cluster_counts.iloc[:, 1:] = 0

    for cluster in clusters:
        cluster_data = df[df[cluster_column] == cluster]
        value_counts = cluster_data[column_to_compare].value_counts()
        cluster_counts.loc[cluster, value_counts.index] = value_counts.values

    cluster_counts.set_index('Cluster', inplace=True)
    cluster_counts.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title(f'Count of {column_to_compare} by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')

    plt.xticks(range(len(clusters)), clusters, rotation=0)

    plt.show()


def select_top_k_features(importances, feature_names, k=5):
    sorted_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    top_feature_names = [feature[0] for feature in sorted_features[:k]]

    return top_feature_names


import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def train_and_evaluate(df, selected_features, viz=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        X = df[selected_features]
        y = df['cluster']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBClassifier()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        confusion_mat = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        if viz:
            print("\nAccuracy Score:", accuracy)

            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=model.classes_, yticklabels=model.classes_)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.show()

        return accuracy
    
def plot_accuracy(df, importances, all_feature):
    accuracies = []
    for k in range(1, len(all_feature) + 1):
        k_features = select_top_k_features(importances, all_feature, k)
        accuracies.append(train_and_evaluate(df, k_features, viz=False))

    x_axes = range(1, len(accuracies) + 1)

    plt.plot(x_axes, accuracies, marker='o')
    plt.xlabel('Number of questions')
    plt.ylabel('Accuracy')
    plt.title('Plot accuracies for each number of questions from most to least important')
    plt.show()