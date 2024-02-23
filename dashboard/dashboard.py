# app.py
import streamlit as st
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ==============
# Model Part
# Sample dataset loading function (replace with your actual dataset)
# For demonstration, we'll generate dummy data
path = os.getcwd()
df = pd.read_pickle(f'/{path}/pretrained_dataset.pkl')

# Define your features and target variable
X = df.drop('wait_time_max', axis=1)
y = df['wait_time_max']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
categorical_features = ['day_of_week', 'entity_description_short', 'weather_main', 'weather_description']
numerical_features = ['attendance', 'nb_units', 'adjust_capacity', 'feels_like', 'humidity', 'rain_1h', 'snow_1h']
binary_features = ['has_night_show', 'has_parade_1', 'has_parade_2']
time_features = ['month', 'hour', 'minutes']

one_hot_encoder = OneHotEncoder()
standard_scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', one_hot_encoder, categorical_features),
        ('num', standard_scaler, numerical_features),
        ('bin', 'passthrough', binary_features),
        ('time', 'passthrough', time_features)
    ]
)

# Decision Tree model pipeline
dt_model = Pipeline([
    ('preprocessor', preprocessor),
#    ('regressor', DecisionTreeRegressor(random_state=42))
    ('regressor', LinearRegression())
])

# Fit the model
dt_model.fit(X_train, y_train)

# ==============
# Streamlit app
# st.title('Wait Time Prediction Tool')
st.header('Attraction Wait Time Predictions', divider='rainbow')

# Side Bar
# Condition Setup
def user_input_features():
    # Collect user inputs
    data = {
        'month': st.sidebar.slider('Month', 1, 12, 1),
        'day_of_week': st.sidebar.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
        'hour': st.sidebar.slider('Hour', 0, 23, 12),
        'minutes': st.sidebar.slider('Minutes', 0, 59, 30),
        'attendance': st.sidebar.number_input('Attendance', value=10000),
        'weather_main': st.sidebar.selectbox('Weather Main', ['Clear', 'Clouds', 'Rain', 'Snow']),
        'weather_description': st.sidebar.selectbox('Weather Description', ['overcast clouds', 'broken clouds', 'sky is clear', 'light rain', 'scattered clouds', 'moderate rain', 'few clouds', 'light snow', 'snow', 'heavy intensity rain']),
        'feels_like': st.sidebar.number_input('Feels Like (°C)', value=20.0, format="%.1f"),
        'humidity': st.sidebar.slider('Humidity (%)', 0, 100, 50),
        'rain_1h': st.sidebar.number_input('Rain Volume (1h)', value=0.0, format="%.1f"),
        'snow_1h': st.sidebar.number_input('Snow Volume (1h)', value=0.0, format="%.1f"),
        'has_night_show': st.sidebar.checkbox('Has Night Show', value=False),
        'has_parade_1': st.sidebar.checkbox('Has Parade 1', value=False),
        'has_parade_2': st.sidebar.checkbox('Has Parade 2', value=False)
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ==============
# Attraction-specific inputs using expanders
attractions = df['entity_description_short'].unique()
attraction_data = {}

# ==============
# Prediction Function
def predict_wait_times(input_df):
    prediction_results = []
    for attraction in attractions:
        temp_input = input_df.copy()
        temp_input['entity_description_short'] = attraction
        temp_input['adjust_capacity'] = attraction_data[attraction]['adjust_capacity']
        temp_input['nb_units'] = attraction_data[attraction]['nb_units']
        predicted_wait_time = dt_model.predict(temp_input)[0]
        prediction_results.append((attraction, predicted_wait_time))
    return prediction_results

# ==============
# Attraction settings form
with st.form("Attraction Settings"):
    # We create a grid layout for attraction settings with 5 columns per row
    cols = st.columns(5)
    for index, attraction in enumerate(attractions):
        # Each attraction setting is placed in a column
        with cols[index % 5]:
            st.markdown(f"**{attraction}**")
            nb_units = st.number_input('Number of Units', value=5, key=f'nb_units_{attraction}')
            adjust_capacity = st.number_input('Adjusted Capacity', value=50, key=f'adjust_capacity_{attraction}')
            # Store each attraction's settings in a dictionary
            attraction_data[attraction] = {'nb_units': nb_units, 'adjust_capacity': adjust_capacity}
    
    # Submit button for the form
    submitted = st.form_submit_button("Submit")
    
# ======
# Number of attractions to display per row
num_per_row = 4

# Function to display predictions in a grid format
def display_predictions_grid(predictions, num_per_row):
    # Calculate the number of rows needed
    num_rows = len(predictions) // num_per_row + (len(predictions) % num_per_row > 0)
    
    # Display each row of predictions
    for i in range(num_rows):
        # Get the subset of predictions for this row
        row_predictions = predictions[i*num_per_row:(i+1)*num_per_row]
        
        # Create columns for each prediction in the row
        cols = st.columns(num_per_row)
        
        # Display each prediction in a column
        for col, prediction in zip(cols, row_predictions):
            with col:
                attraction, wait_time = prediction['Attraction'], prediction['Predicted Wait Time']
                st.markdown(f"**{attraction}**")
                st.metric(label="Predicted Wait Time", value=f"{wait_time:.0f} minutes")


# ==============
# If settings are updated, display the predictions
if submitted:
    prediction_results = predict_wait_times(input_df)
    # Initialize lists to store the attraction names and predicted wait times
    attraction_names = []
    wait_times = []
    
    # Populate the lists with data from prediction results
    for attraction, predicted_wait_time in prediction_results:
        attraction_names.append(attraction)
        wait_times.append(abs(predicted_wait_time))
    
    # Create a DataFrame for the chart
    chart_data = pd.DataFrame({
        'Attraction': attraction_names,
        'Predicted Wait Time': wait_times
    })
    
    # Sort the DataFrame by predicted wait times in descending order
    chart_data = chart_data.sort_values('Predicted Wait Time', ascending=False)
        
    # Split the DataFrame into two halves for the grid display
    first_half = chart_data.iloc[::2].reset_index(drop=True)
    second_half = chart_data.iloc[1::2].reset_index(drop=True)
    
    # Define the number of rows for the grid
    num_rows = max(len(first_half), len(second_half))
        
    # Display the predictions in a grid format with two attractions per row
    st.header(':roller_coaster: Predicted Wait Times', divider='rainbow')
    num_attractions = len(chart_data)
    num_rows = (num_attractions + 1) // 2  # Calculate the number of rows needed
    
    # Loop through each row
    for i in range(num_rows):
        cols = st.columns(2)  # Create two columns
        # Get the index for the attractions in this row
        left_index = i * 2
        right_index = left_index + 1
        
        # Display the attraction name and predicted wait time in the left column
        with cols[0]:
            if left_index < num_attractions:
                attraction_left = chart_data.iloc[left_index]['Attraction']
                wait_time_left = chart_data.iloc[left_index]['Predicted Wait Time']
                st.markdown(f"**{attraction_left}**: :orange[{wait_time_left:.0f}] minutes")
    
        # Display the attraction name and predicted wait time in the right column
        with cols[1]:
            if right_index < num_attractions:
                attraction_right = chart_data.iloc[right_index]['Attraction']
                wait_time_right = chart_data.iloc[right_index]['Predicted Wait Time']
                st.markdown(f"**{attraction_right}**: :orange[{wait_time_right:.0f}] minutes")
                
    # Display the histogram chart
    st.header('Attraction Wait Time Histogram', divider='rainbow')
    st.bar_chart(chart_data.set_index('Attraction'), color="#FF8C42")