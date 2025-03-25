# Quarter-2-Project
This project applies different machine learning approaches to a discretized dataset. The process involves using `discretize.py` on the `ORIGINAL_dataset`, followed by calling various functions to analyze the data using different classification techniques.

1. **Discretization**: Apply `discretize.py` to the `ORIGINAL_dataset`.
2. **Retrieve Data**: Call `get_data()` to obtain the following:
   - `data`: The processed dataset
   - `features`: The features of the dataset
   - `table`: The transformed data table
   - `maxes`: Maximum values of features (if applicable)
3. **Apply Machine Learning Approaches**:
   - **Control (Complement Naive Bayes)**: Use `complement_naive_bayes(data)`.
   - **Approach 1 (Full MCA)**: Use `full_mca(data)`.
   - **Approach 2 (Hierarchical MCA)**: Use `hierarchical_mca(data, features, table, maxes)`.

## Usage
### Running the Project
1. **Discretize the dataset**:
   ```python
   python discretize.py
   ```
2. **Run the main script**:
   ```python
   python main.py
   ```

## Functions
### `get_data()`
Retrieves and returns the processed dataset along with its corresponding attributes.

### `complement_naive_bayes(data)`
Runs the Complement Naive Bayes classification on the dataset.

### `full_mca(data)`
Performs Full Multiple Correspondence Analysis (MCA) on the dataset.

### `hierarchical_mca(data, features, table, maxes)`
Executes Hierarchical MCA, considering dataset features and transformations.



