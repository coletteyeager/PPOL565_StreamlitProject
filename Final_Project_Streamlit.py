import streamlit as st
import pandas as pd
import numpy as np
import requests
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import preprocessing as pp 
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix 
from sklearn.decomposition import PCA

st.set_option('deprecation.showPyplotGlobalUse', False)

# Title
st.title("Census Voting Data")

years = ['1994', '1996', '1998', '2000', '2002', '2004', '2006', '2008', '2010', '2012', '2014', '2016', '2018', '2020']
statedict = {'AL': '1', 'AK': '2', 'AZ': '4', 'AR': '5', 'CA': '6', 'CO': '8', 'CT': '9', 'DE': '10', 'DC': '11', 'FL': '12',
             'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18', 'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 
             'ME':'23', 'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28', 'MO': '29', 'MT': '30', 'NE': '31', 'NV': 
             '32', 'NH': '33', 'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39', 'OK': '40', 'OR': '41', 
             'PA': '42', 'RI': '44', 'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49', 'VT': '50', 'VA': '51', 'WA': 
             '53', 'WV': '54', 'WI': '55', 'WY': '56'}
rvsestatedict = {v: k for k, v in statedict.items()}

# USER INPUT
with st.sidebar:
    st.header("Choose Data to Pull")
    
year = st.sidebar.selectbox(label="Choose a year",
                      options=years)

state = st.sidebar.text_input("Choose a state (type the abbreviation), or view all", value = "")
if (state == ""):
    st.stop()
if (state == 'all' or state == 'All'):
    STATE = '*'
else:
    try:
        STATE = statedict[state]

    except KeyError:
        st.error('Format not accepted or state does not exist. Please try again.')
        st.stop()

# READ IN AND ORGANIZE DATA 

# Read in data from Census API
url = (f"http://api.census.gov/data/{year}/cps/voting/nov")

# Get parameters - some are called different variables in different years
if (year == '1994'):
    param_list = "PES3,PES4,GEMETSTA,GEREG,PERACE,PRHSPNON,PESEX,PRTAGE,PEMARITL,HRNUMHOU,PEAFNOW,PEEDUCA,HUFAMINC,PREXPLF,PRFTLF"
elif(year == '1996' or year == '1998' or year == '2000' or year == '2002'):
    param_list = "PES1,PES2,GEMETSTA,GEREG,PERACE,PRHSPNON,PESEX,PRTAGE,PEMARITL,HRNUMHOU,PEAFNOW,PEEDUCA,HUFAMINC,PREXPLF,PRFTLF"
elif(year == '2004' or year == '2006' or year == '2008'):
    param_list = "PES1,PES2,GTMETSTA,GEREG,PTDTRACE,PEHSPNON,PESEX,PRTAGE,PEMARITL,HRNUMHOU,PEAFNOW,PEEDUCA,HUFAMINC,PREXPLF,PRFTLF"
else:
    param_list = "PES1,PES2,GTMETSTA,GEREG,PTDTRACE,PEHSPNON,PESEX,PRTAGE,PEMARITL,HRNUMHOU,PEAFNOW,PEEDUCA,HEFAMINC,PREXPLF,PRFTLF"

r = requests.get(url,
                params = {"get": param_list,
                         "for": f"state:{STATE}"})

# Create dataframe with data
census_df = pd.DataFrame(data = r.json())
census_df.rename(columns = census_df.iloc[0], inplace = True)
census_df.drop([0], axis = 0, inplace = True)
# Change column names
census_df.columns = ["Voted", "Registered_to_Vote", "Metropolitan",
                     "Geographic_Region", "Race", "Hispanic",
                     "Female", "Age", "Marital_Status", 
                     "Household_Members", "In_Armed_Forces", 
                     "Education_Completed", "Family_Income_category", "Employment_Status", 
                     "Full_Time", "State"]
# Replace number with state abbreviation
census_df.replace({'State': rvsestatedict}, inplace = True)
# Change column types
census_df = census_df.astype({"Voted": int, "Registered_to_Vote": int, "Metropolitan": int, 
                              "Geographic_Region": int, "Race": int, "Hispanic": int, "Female": int,
                              "Age" : int, "Marital_Status": int, "Household_Members": int, 
                              "In_Armed_Forces": int, "Education_Completed": int,
                              "Family_Income_category": int, "Employment_Status": int, "Full_Time": int, "State": str})
col2 = census_df.pop('State')
census_df.insert(0, 'State', col2)

# PREPROCESSING

# Turn some columns into binary or categorical variables - change 2/1 to be 0/1, and make all other values NA (were previously refused response, no response, don't know, or not in universe)
census_df.Voted.replace({2: 0, -1: np.nan, -3: np.nan, -9: np.nan, -2: np.nan},inplace = True)
census_df.Registered_to_Vote.replace({2: 0, -1: np.nan, -3: np.nan, -9: np.nan, -2: np.nan}, inplace = True)
# All people who voted were registered to vote, so need to change any NA's in Registered_to_Vote to 1 where Voted is 1
census_df.loc[census_df.Voted == 1, 'Registered_to_Vote'] = 1
census_df.Female.replace({2: 0},inplace = True)
census_df.Hispanic.replace({2: 0}, inplace = True)
census_df.Race = np.where(census_df.Race > 4, "Other", census_df.Race)
census_df.Race.replace({'1': "White", '2': "Black", '4': "Asian", '3': "American Indian, Aluet, Eskimo"}, inplace = True)
census_df.Marital_Status.replace({2: 1, 3: 0, 4: 0, 5: 0, 6: 0, -1: np.nan},inplace = True)
census_df.In_Armed_Forces.replace({2: 0, -1: np.nan},inplace = True)
census_df.Employment_Status.replace({2: 0, -1: np.nan, 127: np.nan},inplace = True)
census_df.Full_Time.replace({2: 0, -1: np.nan, 127: np.nan},inplace = True)
census_df.Metropolitan.replace({2: 0, 3: np.nan}, inplace = True)
census_df.Geographic_Region.replace({1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}, inplace = True)
census_df.Education_Completed = np.where(census_df.Education_Completed < 39, "No_HS_Diploma", census_df.Education_Completed)
census_df.Education_Completed.replace({'46': "Doctorate", '44': "Masters",
                                       '39': "HighSchool/GED",'42': "Associates", 
                                       '40': "Some_College",'-1': np.nan, '43': "Bachelors",
                                       '45': "Professional_School",'41': "Associates"},inplace = True)
census_df.Family_Income_category.replace({-1: np.nan, -2: np.nan, -3: np.nan}, inplace = True)

# Create different version of family income variable, 1 if greater than $50,000, 0 if less
census_df['Family_Income_dummy'] = np.where(census_df.Family_Income_category > 11, 1, 0)
# Create different version of family income variable that is the middle of the range (or highest of the range in the case of the lowest value, lowest of the range in the case of the highest value)
census_df['Family_Income_actual'] = census_df.Family_Income_category.replace({1: 5000, 2: 6250, 3: 8250, 4: 11250, 5: 13750, 
                                                                           6: 17500, 7: 22500, 8: 27500, 9: 32500, 10: 37500,
                                                                           11: 45000, 12: 55000, 13: 67500, 14: 87500, 
                                                                           15: 125000, 16: 150000})

# Create dummy variables for categorical variables
census_df = pd.get_dummies(census_df, columns = ['Geographic_Region', 'Race', 'Education_Completed'], drop_first = True)
# Drop people younger than 18 - can't vote
census_df = census_df[census_df.Age >= 18]


with st.sidebar:
    st.header("Visualization Options")

dataset = st.sidebar.checkbox("View Dataset")
if dataset:
    rows = st.sidebar.number_input("Number of rows", min_value = 1, max_value = census_df.shape[0], step = 1, value = 5)
    
summary = st.sidebar.checkbox("Summary of Data")



if dataset:
    st.header("Dataset")
    st.dataframe(census_df.head(rows))

if summary:
# Print out info about the data
    st.header("Summary")
    st.write(f"Number of observations: {census_df.shape[0]}")
    st.write(f"Number of features: {census_df.shape[1]}")
    
    st.subheader("Summary of the data features:")
    st.write(census_df.describe())
    
    st.subheader("Missing Data:")
    fig = msno.bar(census_df)
    st.write(fig)
    st.pyplot()
    
    st.subheader("Distribution of Features")
    cols_to_plot = st.sidebar.multiselect("Select features to see histograms", census_df.columns.to_list())
    if st.sidebar.button("View Distributions"):
        census_df[cols_to_plot].hist(figsize = (10, 10))
        plt.show()
        st.pyplot()

machine_learning = st.sidebar.checkbox("Machine Learning Modeling")
if machine_learning:
    # Dependent Variable Selection
    with st.sidebar:
        st.header("Machine Learning")
        dependent = st.selectbox("Choose a dependent variable", options = ("Voted", "Registered to Vote"))
    
    def set_data(dependent, not_dependent):
        # Get rid of whichever variable is not the dependent variable 
        df = census_df.drop([not_dependent], axis = 1)
        # Drop missing values
        df.dropna(inplace = True)
        df.reset_index(drop=True, inplace=True)
        
        X = df.drop([dependent, 'State'], axis = 1)
        y = df[dependent]
        return(X,y)
    
    if dependent == "Voted":
        X, y = set_data('Voted', 'Registered_to_Vote')
        classnames = ['Did Not Vote', 'Voted']
    else:
        X, y = set_data('Registered_to_Vote', 'Voted')
        classnames = [ 'Not Registered', 'Registered']
    
    ### Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=777)

    model = st.sidebar.selectbox("Choose a machine learning algorithm", options = ("Logistic Regression", "SVM", "Decision Tree", 
                                                                                   "KNN", "Compare all four"))
    n_neighbors = range(1,11)
    depths = range(1,11)
    c_options = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    
    def plot_validation(train_acc, test_acc, parameters, parameter_name, lab):
        plt.plot(parameters, train_acc, label="training accuracy")
        plt.plot(parameters, test_acc, label="test accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel(f"{parameter_name}")
        plt.title(f'Validation Curve for {lab} Model')
        plt.legend()
        plt.show()
        st.pyplot()
    
    if model == "Compare all four":
        # Create Pipeline to compare all methods
        pipe = Pipeline([('model',None)])
        search_space = [
            # Logistic Regression
            {'model' : [LogisticRegression()],
             'model__C': c_options},
            
            # SVM
            {'model' : [SVC()],
             'model__C': c_options},
    
            # KNN with K tuning param
            {'model' : [KNeighborsClassifier()],
             'model__n_neighbors': n_neighbors},
    
            # Decision Tree with the Max Depth Param
            {'model': [DecisionTreeClassifier()],
             'model__max_depth': depths},
        ]
        # (5) Put it all together in the grid search
        search = GridSearchCV(pipe, search_space, scoring='accuracy', cv = 5)
        search.fit(X_train,y_train)
        st.write("The best model is:", search.best_params_)
        st.write(f"The highest training accuracy is: {round(search.best_score_, 4)}")
        st.write(f"Test score: {round(search.score(X_test, y_test), 4)}")
        st.stop()
    elif model == "Logistic Regression":
        param_name = "C" 
        param_number = st.sidebar.select_slider("Choose regularization value for C", options = c_options, value = 1.0)
        mod = LogisticRegression(C = param_number)
        train_acc = [LogisticRegression(C=each).fit(X_train, y_train).score(X_train, y_train) 
                      for each in c_options]
        test_acc = [LogisticRegression(C=each).fit(X_train, y_train).score(X_test, y_test) 
                      for each in c_options]
        if st.sidebar.checkbox("View Validation Curve"):
            st.subheader("Comparing Parameters")
            plot_validation(train_acc, test_acc, c_options, "C", "Logistic Regression")
    elif model == "SVM":
        param_name = "C" 
        param_number = st.sidebar.select_slider("Choose regularization value for C", options = c_options, value = 1.0)
        mod = SVC(C = param_number)
        train_acc = [SVC(C=each).fit(X_train, y_train).score(X_train, y_train) 
                      for each in c_options]
        test_acc = [SVC(C=each).fit(X_train, y_train).score(X_test, y_test) 
                      for each in c_options]
        if st.sidebar.checkbox("View Validation Curve"):
            st.subheader("Comparing Parameters")
            plot_validation(train_acc, test_acc, c_options, "C", "SVM")
    elif model == "Decision Tree":
        param_name = "Max Depth" 
        param_number = st.sidebar.slider("Choose a number for the max depths of the tree", 1, 10, value = 4)
        mod = DecisionTreeClassifier(max_depth = param_number)
        train_acc = [DecisionTreeClassifier(max_depth=each).fit(X_train, y_train).score(X_train, y_train) 
                      for each in range(1,11)]
        test_acc = [DecisionTreeClassifier(max_depth=each).fit(X_train, y_train).score(X_test, y_test) 
                      for each in range(1,11)]
        if st.sidebar.checkbox("View Validation Curve"):
            st.subheader("Comparing Parameters")
            plot_validation(train_acc, test_acc, range(1,11), "Max Depth", "Decision Tree")
    elif model == "KNN":
        param_name = "Neighbors" 
        param_number = st.sidebar.slider("Choose a number of neighbors", 1, 10, value = 5)
        mod = KNeighborsClassifier(n_neighbors = param_number)
        train_acc = [KNeighborsClassifier(n_neighbors=each).fit(X_train, y_train).score(X_train, y_train) 
                      for each in n_neighbors]
        test_acc = [KNeighborsClassifier(n_neighbors=each).fit(X_train, y_train).score(X_test, y_test) 
                      for each in n_neighbors]
        if st.sidebar.checkbox("View Validation Curve"):
            st.subheader("Comparing Parameters")
            plot_validation(train_acc, test_acc, n_neighbors, "Neighbors", "KNN")
    mod.fit(X_train, y_train)
    if st.sidebar.checkbox("View Accuracy Scores"):
        # Fit model and get accuracy for specific parameters  
        y_pred = mod.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.subheader(f"Accuracy Scores for {param_name} = {param_number}")
        st.write(f"Training score: {round(mod.score(X_train, y_train), 4)}")
        st.write(f"Test score: {round(mod.score(X_test, y_test), 4)}")
    
        scores = cross_val_score(mod, X_test, y_test, cv=5)
        st.write(f"Mean CV score: {round(scores.mean(), 4)}")
    
        st.write(f"Recall score {round(recall_score(y_test, y_pred), 4)}")
    
        st.write("Confusion Matrix:")
        st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred),
                columns=["Predicted negative", "Predicted positive"],
                index=["Actual negative","Actual positive"]))
    
    if model == "Decision Tree":
        if st.sidebar.checkbox("View Decision Tree"):
            st.subheader("Decision Tree Visualization")
            fig = plt.figure(figsize = (100, 35))
            _ = tree.plot_tree(mod, 
                               feature_names = X.columns,
                               class_names = classnames,
                               filled = True,
                               impurity = True,
                               fontsize = 70)
            st.pyplot()
         
        
   
        
        
            
    
    
    
             

