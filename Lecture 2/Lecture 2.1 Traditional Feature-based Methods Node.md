# Traditional Feature-based Methods: Node

![Traditional ML Pipeline](Traditional ML Pipeline.png)

![Traditional ML Pipeline2](Traditional ML Pipeline2.png)



## Feature Design

Focus on **undirected graphs**

+ Node-level prediction
+ Link-level prediction
+ Graph-level prediction



## Machine Learning in Graphs

Given:
$$
G=(V,E)
$$
Learn a function:
$$
f:V \to R
$$
How do we learn the function?



## Node-level Tasks

+ Node classification
+ Goal: characterize the structure and position of a node in the network
  + Node degree
  + Node centrality
  + Clustering coefficient
  + Graphlets
+ Degree
  + ![Node Degree](Node Degree.png)
+ Centrality
  + Takes the **node importance in a graph** into account
  + Different ways to model importance:
    + Eigenvector centrality
      + ![Eigenvector centrality](Eigenvector centrality.png)
      + ![Eigenvector centrality2](Eigenvector centrality2.png)
    + Betweenness centrality
      + ![Betweenness centrality](Betweenness centrality.png)
      + ![Closeness centrality](Closeness centrality.png)
+ Clustering Coefficient
  + ![Clustering Coefficient](Clustering Coefficient.png)

+ Graphlets
  + ![Graphlets](Graphlets.png)
  + counting triangles(social network, friend in common)
  + ![Graphlets2](Graphlets2.png)
  + ![GDV](GDV.png)
  + Similar to chemical fingerprints(not only 0/1, but counts)
  + **Induced graph**
  + As a signature of a node that describes the topology of node's neighborhood
  + Provides a measure of a node's local network topology

