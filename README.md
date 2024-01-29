g# Learning Machine Learnings and Deep Learningsnnv

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


**7.Day_7 of Learning:**
- **Linear Discriminant Analysis:** In this analysis, I standardized features like 'frontal_lobe' and 'rear_width' using Z-score normalization on a crab dataset and applied Linear Discriminant Analysis (LDA) in both two and three dimensions. The resulting components were integrated back for visualization through a 2D scatter plot and a 3D plot. Simultaneously, on a digit image dataset, I used Fisher Discriminant Analysis (FDA), or LDA, to reduce dimensionality to two and three dimensions, visualizing the results through 2D and 3D scatter plots. The clear organization of steps and well-chosen variable names enhances understanding of both dimensionality reduction techniques and their visual outcomes.

[Linear Discriminant Analysis](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/tree/main/06.%20Linear%20Discriminant%20Analysis%20And%20Visualization)
- **01. LDA Analysis And Visualization on Tabular Data:**
                                                         In this analysis, I began by loading a crab dataset from a CSV file and standardized key features, including 'frontal_lobe' and 'rear_width,' using Z-score normalization. Subsequently, I applied Linear Discriminant Analysis (LDA) for dimensionality reduction, initially in two dimensions ('LDA1' and 'LDA2') and then extended to three dimensions ('LDA1', 'LDA2', and 'LDA3'). The resulting LDA components were integrated back into the dataset. To visually assess the discriminative power of LDA, I generated a 2D scatter plot, color-coding data points by crab class, and a 3D plot representing the distribution of data in the three-dimensional space defined by the LDA components. 

  Link:
  [01. LDA Analysis And Visualization on Tabular Data](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/tree/main/06.%20Linear%20Discriminant%20Analysis%20And%20Visualization/01.%20LDA%20And%20Visualization%20on%20Tabular%20Data)

  ![Image](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/blob/main/Images/LDA-on-Tabular.jpg)

- **02. LDA Analysis And Visualization on Image Data:**
                                                         In this analysis, Fisher Discriminant Analysis (FDA), also known as Linear Discriminant Analysis (LDA), i applied it  to a dataset containing handwritten digit images from sklearn. The imported a single digit image ('2') and created a visual grid of digit examples. Employing FDA, the dataset's dimensionality was reduced to two ('FDA1' and 'FDA2') and three dimensions ('FDA1', 'FDA2', and 'FDA3'). The resulting FDA components were converted into a DataFrame for visualization. A 2D scatter plot color-coded by digit classes and a 3D scatter plot were generated, providing a visual representation of how FDA separates different digit classes in reduced dimensions. 

  Link:
  [02. LDA Analysis And Visualization on Image Data](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/tree/main/06.%20Linear%20Discriminant%20Analysis%20And%20Visualization/02.%20LDA%20And%20Visualization%20on%20Image%20Data)

![Image](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/blob/main/Images/LDA-on-Image.jpg)



**8.Day_8 of Learning:**
- **Linear Regression Analysis**: In this day, I started by importing a life expectancy dataset starting with preprocessing the data along with handling missing data, converting categorical columns to the appropriate type, and splitting the data into training and testing sets. Utilizing linear regression, both from scikit-learn and statsmodels, I fitted a model to predict life expectancy based on various features. The coefficients and intercept of the model were examined to understand their significance. Then for model diagnostics residual analysis, probability plots, and quantile-quantile (Q-Q) plots was made to assess the assumptions of linear regression. Visualizations were generated to evaluate model performance. The mean squared error and normalized mean squared error were computed to quantify the model's accuracy on the training set, along with the R-squared value.
  
Link:
[Linear Regression Analysis](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/tree/main/07.%20Linear%20Regression%20Analysis%20And%20Visualization)

![Image](https://github.com/ranzeet013/Machine_Learnings_Deep_Learnings/blob/main/Images/Linear-Regression.png)


































