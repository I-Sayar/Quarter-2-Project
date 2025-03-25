import pandas as pd
import prince  # Library for MCA and CA
import numpy as np
import matplotlib.pyplot as plt


# Step 1: Create a small dataset
data = pd.read_csv("heart_attack_youth_adult_france.csv")

'''data = {
    "Favorite Color": ["Red", "Blue", "Green", "Blue", "Red"],
    "Preferred Pet": ["Dog", "Cat", "Dog", "Fish", "Dog"],
    "Hobby": ["Reading", "Sports", "Painting", "Reading", "Sports"]
}
df = pd.DataFrame(data)

# Step 2: Initialize and fit MCA
mca = prince.MCA(n_components=2, random_state=42)
mca = mca.fit(df)

# Step 3: Transform the data
mca_transformed = mca.transform(df)

# Display results
print("MCA Eigenvalues:")
print(mca.eigenvalues_)

print("\nMCA Transformed Data:")
print(mca_transformed)'''




data = pd.read_csv("heart_attack_youth_adult_france.csv")


np.random.seed(42)
sampled_data = data.sample(n=10000)





# Initialize the MCA model
mca = prince.MCA(n_components=6, random_state=42)

# Fit the MCA model to the sampled dataset
mca_result = mca.fit(sampled_data)

# Transform the data using the MCA model
mca_transformed = mca.transform(sampled_data)

# Optional: Save the transformed dataset to a CSV file
mca_transformed.to_csv('mca_transformed_data.csv', index=False)


