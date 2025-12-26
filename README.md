# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the customer dataset from the CSV file and select annual income and spending score as features.
2. Apply the K-Means clustering algorithm with a fixed number of clusters.
3. Assign each customer to a cluster based on similarity.
4. Visualize the formed clusters and their centroids for customer segmentation.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Joshua Abraham Philip A
RegisterNumber: 25013744

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("Mall_Customers.csv")

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(8,6))
for i in range(3):
    plt.scatter(
        X[df['Cluster'] == i]['Annual Income (k$)'],
        X[df['Cluster'] == i]['Spending Score (1-100)'],
        label=f'Cluster {i+1}'
    )

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=200,
    c='yellow',
    label='Centroids',
    marker='X'
)

plt.title('Customer Segmentation (K-Means)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

print(df)


*/
```

## Output:
<img width="637" height="729" alt="image" src="https://github.com/user-attachments/assets/5aa37f3c-ebc9-43b2-8529-a0af0ead34ad" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
