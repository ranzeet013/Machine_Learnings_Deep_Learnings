

# Crab Dataset Visualization and Analysis

## Overview

This project involves the analysis and visualization of a dataset containing information about crabs. The dataset ('crabs.csv') is loaded and processed using Python, utilizing popular data science and visualization libraries such as pandas, numpy, seaborn, and matplotlib.

## Table of Contents

- [Getting Started](#getting-started)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [License](#license)

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ranzeet013/crab-dataset-visualization-and-analysis.git
   cd crab-dataset-visualization-and-analysis
   ```

2. **Install Dependencies:**
   ```bash
   pip install pandas numpy seaborn matplotlib
   ```

3. **Run the Script:**
   ```bash
   python Dataset Inspection.py
   ```

## Usage

The Python script (`Dataset Inspection.py`) performs the following tasks:

- Loads the dataset ('crabs.csv') into a pandas DataFrame.
- Renames columns for clarity and maps categorical values.
- Creates a new column ('class') by concatenating values from other columns.
- Generates descriptive statistics and visualizations, including box plots, histograms, and pair plots.

## Features

- **Data Loading and Preprocessing:**
  - Efficiently loads and preprocesses the crab dataset.
  - Renames columns and maps categorical values.

- **Descriptive Statistics:**
  - Displays descriptive statistics for specific columns.

- **Data Visualization:**
  - Creates box plots to visualize feature distribution by class.
  - Generates histograms for individual feature distribution.
  - Provides comprehensive pair plots for feature relationships.

- **Code Readability:**
  - Well-commented code with explanations for each step.
  - Follows PEP 8 naming conventions for variable names.

- **Flexibility:**
  - Designed to be flexible for easy adaptation to other datasets.

## Dependencies

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [seaborn](https://seaborn.pydata.org/)
- [matplotlib](https://matplotlib.org/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.






