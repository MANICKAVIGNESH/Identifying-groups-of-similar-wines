# Identifying-groups-of-similar-wines
Assignment: identifying groups of similar wines

Overview
This project implements a clustering algorithm to group similar wines based on their features. The algorithm uses a custom matrix class to calculate distances and manage data. The core of the project includes functionalities for standardizing data, calculating Euclidean and weighted distances, and updating cluster centroids iteratively.

Features
Standardization of Data: The project standardizes the dataset to ensure that all features contribute equally to distance calculations.
Distance Calculation: It computes both Euclidean and weighted distances between data points and cluster centroids.
Dynamic Weight Adjustment: The algorithm adjusts weights dynamically based on the separation within and between clusters.
Iterative Clustering: The clustering process runs iteratively to refine cluster assignments and centroids.
Requirements
To run this project, you need the following libraries:

numpy
pandas
You can install the required libraries using pip:

bash
Copy code
pip install numpy pandas
Running the Code
Clone or download this repository to your local machine.
Make sure you have the required dataset in CSV format.
Update the dataset file path in the run_test() function. See the instructions below.
Run the script. If you're using Google Colab, ensure the necessary files are accessible from your Google Drive.
Updating the Dataset File Path
If you have a different dataset that you would like to use, please update the file path in the code. Locate the following line in the run_test() function:

python
Copy code
m = Matrix('/content/drive/MyDrive/Company Project/Anubavam/Data (2).csv')
Replace the file path with the path to your dataset file. Make sure the file is accessible from your environment (e.g., Google Drive) before running the code.

For example, if your dataset is named my_data.csv and is located in a different folder, update the line as follows:

python
Copy code
m = Matrix('/content/drive/MyDrive/YourFolder/my_data.csv')
Ensure that the file format is compatible with the code (CSV format) for successful execution.

Example Output
Upon successful execution, the script will output the initial centroids and the updated cluster assignments for each iteration, along with the frequency of each cluster:

plaintext
Copy code
Initial Centroids: [[...]]
Updated Cluster Assignments: [[...]]
2 = {1.0: 33, 2.0: 24, ...}
3 = {1.0: 20, 2.0: 15, ...}
...
Code Structure
The main class in the project is Matrix, which handles operations related to matrix manipulation, distance calculation, and clustering. Here’s a brief overview of the methods in the Matrix class:

__init__(self, filename=None): Initializes the matrix and loads data from a CSV file if provided.
load_from_csv(self, filename): Loads data from a specified CSV file into the matrix.
standardize(self): Standardizes the matrix data.
get_distance(self, other_matrix, row_i): Calculates Euclidean distance between a specified row and all rows in another matrix.
get_weighted_distance(self, other_matrix, weights, row_i): Calculates weighted distances.
get_count_frequency(self): Computes frequency counts of unique values in the matrix.
Various static methods for centroid initialization, separation calculations, and weight updates.
Conclusion
This project serves as an effective demonstration of clustering algorithms, showcasing how data points can be grouped based on their features. By following the steps outlined above, you can easily adapt the code to work with your dataset and explore the clustering capabilities.
