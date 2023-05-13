# Project Overview

Breast cancer is a serious and often life-threatening disease that affects millions of people worldwide. Early detection and accurate diagnosis are crucial for successful treatment, making it important to identify the most informative features for predicting the type of breast cancer tumor. This project is a feature selection analysis of the Breast Cancer dataset, which aims to demonstrate the impact of different feature selection techniques on the accuracy of a RandomForestClassifier trained on the dataset.

### Dataset
The Breast Cancer dataset used in this project is available in the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)). It contains information about 569 breast cancer tumors described by 30 features such as radius, texture, and perimeter of each tumor. The target variable indicates whether the tumor is malignant or benign.


### Feature Selection Techniques
The following feature selection techniques are implemented in this project:

**Correlation-based Feature Selection**: a filter-based method that uses the correlation between features and the correlation between features and the target variable, to select the most informative features.

**Filter-based Feature Selection**: Methods which use statistical tests to independently assess features and identify the most important ones.

**Wrapper-based Feature Selection**: Methods that evaluate subsets of features using a machine learning algorithm and selects the best subset based on performance.

**Embedded Feature Selection**: Methods that use a machine learning algorithm's intrinsic feature selection. An example of this is L1 regression.

### Usage

To use this project, 

- First clone the repository:

```bash
git@github.com:Aldion0731/feature-selection-breast-cancer-dataset.git
```

- Second, install requirements

```bash
pipenv sync
```

- Finally, navigate to `src/notebooks/feature_selection.ipynb` and run the code in the notebook.


### Results

![Results](/results/results.PNG)

### Conclusion

The results of this project demonstrate that not only are feature selection techniques useful for reducing resource requirements and model complexity, but if properly done can also improve model accuracy by finding the best subset of features for predicting the target variable. Whereas we did not see significant improvements in recall, the fact that we were able to see improvements in metrics with less complex models presents a compelling case for feature selection methods.

Additional feature selection techniques and machine learning algorithms can be explored to further improve performance. Experimentation with the parameters use for the techniques employed in this project may also improve results.