# Learning Machine Learnings and Deep Learnings

## Journey Overview 
I'll be sharing everything I learn on my journey of learning machine learning and deep learning. From basics to advanced concepts, I'll cover essential tools and practical applications. Whether you're a beginner or looking to deepen your understanding, my posts will be a helpful resource, documenting the learning process with hands-on experiences in machine learning and deep learning.

**1.Day_1 of Learning:**
- **Data Inspecting**: Embarking on the journey of learning machine learning and deep learning, I initiated the process with an analysis of a dataset related to crabs. Key preprocessing steps included refining column names and mapping categorical values in the 'species' and 'sex' columns. The introduction of a new 'class' column by amalgamating values from 'species' and 'sex' was followed by presenting concise descriptive statistics and visualizations. This foundational analysis serves as a pivotal step in my pursuit of mastering machine learning and deep learning concepts.

Link:
[Data Inspection](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/tree/main/00.%20Data%20Inspection)

![Image](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/blob/main/00.%20Data%20Inspection/Dataset/DataInspection2.png)

**2.Day_2 of Learning:**
- **Principal Component Analysis:** Principal Component Analysis (PCA) is a dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional space. It identifies orthogonal axes, called principal components, capturing the maximum variance in the data. PCA utilizes eigenvalues and eigenvectors to determine these components, resulting in uncorrelated variables. On the second day of exploring machine learning and data analysis, i focused on implementing Principal Component Analysis (PCA) for dimensionality reduction on a crab dataset. After standardizing numerical features and conducting PCA, the script presented key metrics like explained variance ratios and cumulative explained variance. Visualizations, including scree plots and heatmaps, offered insights into the dataset's structure. The transformed data was utilized to create 2D and 3D scatter plots, providing a visual representation of patterns. This day's work significantly contributed to understanding advanced techniques for data exploration and dimensionality reduction using PCA.

Link:
[Principal Component Analysis](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/tree/main/01.%20Principal%20Component%20Analysis%20And%20Visualization)

![Image](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/blob/main/Images/PCA.jpg)

**3.Day_3 of Learning:**
- **t-Stochastic Neighbor Embedding Analysis:** In my analysis of a cleaned crab dataset, I employed various preprocessing techniques to enhance interpretability. Initially, I applied Min-Max Scaling to ensure standardized numerical values for specific features like 'frontal_lobe' and 'rear_width.' Subsequently, I standardized the dataset using Z-score normalization (Standard Scaling), transforming selected features to have a mean of 0 and a standard deviation of 1. After these preprocessing steps, I utilized t-distributed Stochastic Neighbor Embedding (t-SNE) for efficient dimensionality reduction and visualization. The application of t-SNE generated both 2D and 3D scatter plots, unveiling intricate patterns and clusters within the high-dimensional data. The color-coded 2D plot highlighted distinct crab classes, while the 3D visualization added depth and complexity to the exploration.

[t-Stochastic Neighbour Embedding](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/tree/main/02.%20t-Stochastic%20Neighbor%20Embedding%20Analysis%20And%20Visualization)
- **01. t-SNE Analysis And Visualization on Raw Data:**
                                                         I applied t-distributed Stochastic Neighbor Embedding (t-SNE) to a cleaned crab dataset for efficient dimensionality reduction and visualization. Using key columns such as 'frontal_lobe,' 'rear_width,' and others, I generated both 2D and 3D scatter plots, revealing intricate patterns and clusters within the high-dimensional data. The color-coded 2D plot highlighted distinct crab classes, while the 3D visualization added depth and complexity to the exploration. These t-SNE visualizations serve as insightful tools for quickly interpreting the dataset's inherent structures and relationships.

  Link:
  [01. t-SNE Analysis And Visualization on Raw Data](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/tree/main/02.%20t-Stochastic%20Neighbor%20Embedding%20Analysis%20And%20Visualization/01.%20t-SNE%20Analysis%20And%20Visualization%20on%20Raw%20Data)

- **02. t-SNE Analysis And Visualization on Scaled Data:**
                                                           I first applied Min-Max Scaling to a cleaned crab dataset, ensuring standardized numerical values for specific features. Subsequently, I employed t-distributed Stochastic Neighbor Embedding (t-SNE) to visualize the scaled data in both 2D and 3D formats. The resulting scatter plots reveal intricate structures and clusters within the dataset, with color-coded distinctions highlighting different crab classes. This combined approach of scaling and t-SNE visualization enhances the interpretability of the dataset, providing valuable insights into its underlying patterns and relationships.

  Link:
  [02. t-SNE Analysis And Visualization on Scaled Data](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/tree/main/02.%20t-Stochastic%20Neighbor%20Embedding%20Analysis%20And%20Visualization/02.%20t-SNE%20Analysis%20And%20Visualization%20on%20Scaled%20Data)

- **03. t-SNE Analysis And Visualize on Standard Data:**
                                                         I first standardized a cleaned crab dataset using Z-score normalization (Standard Scaling). The selected features, including 'frontal_lobe,' 'rear_width,' and others, were transformed to have a mean of 0 and a standard deviation of 1. Subsequently, t-distributed Stochastic Neighbor Embedding (t-SNE) was applied to generate 2D and 3D visualizations. The resulting scatter plots effectively showcase intricate structures and clusters within the standardized data, with color-coded distinctions for different crab classes. This combined approach of standardization and t-SNE visualization enhances the interpretability of the dataset, offering valuable insights into its inherent patterns and relationships.

  Link:
  [03. t-SNE Analysis And Visualize on Standard Data](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/tree/main/02.%20t-Stochastic%20Neighbor%20Embedding%20Analysis%20And%20Visualization/03.%20t-SNE%20Analysis%20And%20Visualize%20on%20Standard%20Data)

  
![Image](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/blob/main/Images/t-SNE-2d.jpg)
![Image](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/blob/main/Images/t-SNE-3d.jpg)

**4.Day_4 of Learning:**
- **Locally Linear Embedding Analysi:** 
On the fourth day of the analysis, I continued exploring the cleaned crab dataset, focusing on advanced techniques for dimensionality reduction and visualization. I implemented the Locally Linear Embedding (LLE) algorithm with two different configurations: first, with 2 dimensions, and later, with 3 dimensions. The application of LLE aimed to capture nonlinear structures in the data while preserving local relationships between data points. After successfully embedding the dataset into lower-dimensional spaces, I created visualizations, including 2D scatter plots and a 3D scatter plot. These visualizations provided deeper insights into the complex patterns and clusters within the high-dimensional data.

Link:
[Locally Linear Embedding Analysi](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/tree/main/03.%20Locally%20Linear%20Embedding%20Analysis%20And%20Visualization)

![Image](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/blob/main/Images/LLE.jpg)


**5.Day_5 of Learning:**
- **Multidimensional Scaling Analysis:** On the fifth day of analysis, I utilized a cleaned crab dataset and implemented preprocessing techniques to enhance interpretability. Initially, I applied Min-Max Scaling to standardize numerical features such as 'frontal_lobe' and 'rear_width,' ensuring a consistent scale for all variables. Subsequently, I employed Multidimensional Scaling (MDS) as a dimensionality reduction technique. The MDS algorithm, executed with two and three dimensions, aimed to capture the dissimilarities between data points while preserving their relationships. The resulting MDS components were incorporated into the dataset. The stress values, indicative of how well the reduced representation preserved original distances, were assessed for both MDS configurations. Visualizations, including 2D and 3D scatter plots, were created to explore the intrinsic structures and clusters within the data.

Link:
[Multidimensional Scaling Analysis](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/tree/main/04.%20Multidimensional%20Scaling%20Analysis%20and%20Visualization)

![Image](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/blob/main/Images/MDS.jpg)

**6.Day_6 of Learning:**
- **Isometric Mapping Analysis:** On this day of analysis, I implemented Min-Max Scaling on numerical features such as 'frontal_lobe' and 'rear_width,' ensuring a standardized scale across all variables. Subsequently, I employed Isometric Mapping (ISOMAP), a dimensionality reduction technique, to capture the inherent geometry of the data. ISOMAP was applied with both two and three dimensions, aiming to preserve dissimilarities between data points while maintaining their relationships. The resulting ISOMAP components were integrated back into the dataset. I assessed the stress values for both ISOMAP configurations to gauge the accuracy of the reduced representation in preserving the original distances. Then, I created visualizations, including 2D and 3D scatter plots, to visually explore the intrinsic structures and clusters within the dataset.

Link:
[Isometric Mapping Analysis](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/tree/main/05.%20Isometric%20Mapping%20Analysis%20And%20Visualization)

![Image](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/blob/main/Images/ISOMAP.jpg)











