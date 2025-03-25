import pandas as pd



#THIS IS THE CODE USED TO DISCRETIZE THE DATA  

dataset = pd.read_csv("Discretized_HeartAttack.csv")
#HAVE TO COMMENT OUT EVERYTHING BECAUSE IT IS A ONE TIME ACTION. THE DATASET IS MODIFIED ONCE THEREFORE UNCOMMENTING -> ERRORS 

#combine both blood pressures into one attribute after discretizing them. 
#also explain how systolic BP is better indicator for heart attacks? and how this affects older people more than younger people?

def bloodPressureDiscretize(row): 
    blood_pressure_systolic = row['Blood_Pressure_Systolic']
    blood_pressure_diastolic = row['Blood_Pressure_Diastolic']

    if (blood_pressure_systolic) < 120 and (blood_pressure_diastolic) < 80: 
        return 'Normal'
    if (120 <= blood_pressure_systolic <= 129) and (blood_pressure_diastolic< 80): 
        return 'Elevated'
    if (130 <= blood_pressure_systolic <= 139) or (80 <= blood_pressure_diastolic <= 89): 
        return 'High_BP_Stage1'
    if (blood_pressure_systolic >= 140) or (blood_pressure_diastolic >=90): 
        return 'High_BP_Stage2'
    if(blood_pressure_systolic>180) and (blood_pressure_diastolic>120): #can just delete this?
        return 'Hypertensive_Crisis'
    if(blood_pressure_systolic>180) or (blood_pressure_diastolic>120): 
        return 'Hypertensive_Crisis'

#dataset['BP'] = dataset.apply(bloodPressureDiscretize, axis=1)

#check for any missing values: 
'''missingVals = dataset['BP'].isnull() 
if missingVals.any(): 
    print(dataset[missingVals])

col = dataset.pop('BP')  
dataset.insert(6, 'BP', col)'''

#dataset = dataset.drop(['Cholesterol', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic'], axis=1)


cholesterol_level = dataset.Cholesterol_Level
bins = [-float('inf'), 199, 239, float('inf')]  # Ranges: <200, 200-239, >=240
labels = ['Healthy', 'At_Risk', 'Dangerous']
#dataset["Cholesterol_Level"] = pd.cut(cholesterol_level, bins=bins, labels=labels, include_lowest=True)

air_pollution_level = dataset.Air_Pollution_Level 
bins = [0, 50, 100, 150, 200, 300, float('inf') ] #Ranges: 0-50, 51-100, 101-150, 151-200, 201-300, >=301 
labels = ['Good', 'Moderate', 'Unhealthy_Sensitive', 'Unhealthy', 'Very_Unhealthy', 'Hazardous']
#dataset["Air_Pollution_Level"] = pd.cut(air_pollution_level, bins=bins, labels=labels, include_lowest=True)

bmi = dataset.BMI
bins = [0, 18.5, 25, 30, float('inf')] #ranges: 0-18.5, 18.5 < x < 25, 25 =< x < 30, >= 30 
labels = ['underweight', 'healthy_weight', 'overweight', 'obesity']
#dataset["BMI"] = pd.cut(bmi, bins=bins, labels=labels, include_lowest=True) 

#dataset = dataset.drop(['Age', 'Weight_kg'], axis=1) Don't need 


def genderHeightClassification(row): 
    gender = row['Sex']
    height = row['Height_cm']


    if gender=='Female': 

        if height < 160: return 'Short'
        if 160 <= height <= 170: return 'Average'
        if 170 < height <= 180: return 'Tall'
        return 'Very_Tall'
    
    if height < 165: return 'Short'
    if 165 <= height < 176: return 'Average'
    if 176 <= height < 190: return 'Tall'
    return 'Very_Tall'

#dataset['Height'] = dataset.apply(genderHeightClassification, axis = 1)

alcohol_lvl = dataset.Alcohol_Consumption

def alcConsumpDiscretize(value):
    if value == 0:
        return "No Alcohol Consumption"
    elif 1 <= value <= 4:
        return "Low Consumption"
    elif 5 <= value <= 9:
        return "Moderate Consumption"
    elif 10 <= value <= 15:
        return "High Consumption"
    return "Very High Consumption"

#dataset['Alcohol_Consumption'] = dataset['Alcohol_Consumption'].apply(alcConsumpDiscretize)


def physicalActivityDiscretize(hours):
    if hours == 0:
        return "No Activity"
    elif 0 < hours <= 3.7:
        return "Low Activity"
    elif 3.7 < hours <= 7.5:
        return "Moderate Activity"
    elif 7.5 < hours <= 11.3:
        return "Active"
    return "Very Active"


# Apply the categorization function to the 'Physical_Activity_Hours' column
#dataset['Physical_Activity_Hours'] = dataset['Physical_Activity_Hours'].apply(physicalActivityDiscretize)


def stressLevelDiscretize(level):
    if 1 <= level <= 3:
        return "Low Stress"
    elif 4 <= level <= 5:
        return "Moderate Stress"
    elif 6 <= level <= 8:
        return "High Stress"
    elif 9 <= level <= 10:
        return "Very High Stress"


#dataset['Stress_Level'] = dataset['Stress_Level'].apply(stressLevelDiscretize)

def heartRateDiscretize(rate):
    if 50 <= rate <= 59:
        return "Low Heart Rate (Bradycardia)"
    elif 60 <= rate <= 100:
        return "Normal Heart Rate"
    elif 101 <= rate <= 110:
        return "Elevated Heart Rate"
    elif 111 <= rate <= 119:
        return "High Heart Rate (Tachycardia)"

# Apply the categorization function
#dataset['Heart_Rate'] = dataset['Heart_Rate'].apply(heartRateDiscretize)

#dataset = dataset.drop('Patient_ID', axis=1)

dataset["Heart_Attack"] = dataset.pop("Heart_Attack")

#update the dataset 
dataset.to_csv("heart_attack_youth_adult_france.csv", index=False) #have to include index = False or will add a new column to DS every run



