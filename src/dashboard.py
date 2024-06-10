from data.get_data import get_image_path, get_pickle_data
from utils import query_model, encode_categorical_columns, decode_categorical_columns
from charts import show_bar_chart, show_error_chart, show_grouped_bar_chart
import streamlit as st
import pandas as pd
import numpy as np

# Load data
df = get_pickle_data('dataframe')
metrics_scores = get_pickle_data('metrics_scores')

models = {
    "Random Forest": get_pickle_data('randomForest_model'),
    "Gradient Boosting": get_pickle_data('gradientBoosting_model'),
    "Support Vector Machine": get_pickle_data('supportVectorRegression_model'),
    "Linear Regression": get_pickle_data('linearRegression_model'),
}

# Streamlit UI
st.title('Real Estate Price Prediction')
st.sidebar.image(get_image_path('logo.png'), use_column_width=True)

# Query form
st.sidebar.header('Query Model')
location_input = st.sidebar.text_input('Location', '')

# Get the maximum and minimum values of the area from the DataFrame
max_area = df['Area'].max()
min_area = df['Area'].min()

area_range_input = st.sidebar.slider('Area Range', min_value=float(min_area), max_value=float(max_area), value=(float(min_area), float(max_area)))

rooms_array = np.sort(df['Rooms'].dropna().unique())
floor_array = np.sort(df['Floor'].dropna().unique())
rooms_array = np.insert(rooms_array, 0, 'Wszystkie opcje')
floor_array = np.insert(floor_array, 0, 'Wszystkie opcje')

rooms_input = st.sidebar.selectbox('Number of Rooms', rooms_array, index=0)
floor_input = st.sidebar.selectbox('Floor', floor_array, index=0)

model_input = st.sidebar.selectbox('Select Model', list(models.keys()))

submit_button = st.sidebar.button('Submit')

# Create tabs
tab1, tab2, tab3 = st.tabs(["Query Results", "Entire Dataset", "Prediction statistics"])

with tab1:
    st.header('Data Visualization for Searched Query')
    if submit_button:
        st.subheader('Query Results')
        result_df = query_model(df, models[model_input], Location=location_input, Rooms=rooms_input, Floor=floor_input, Area_range=(float(min_area), float(max_area)))
        if result_df is not None:
            st.write(result_df)

            # Show prediction errors
            st.header('Error Analysis for Query Results')
            result_df['Error'] = result_df['Price'] - result_df['Predicted Price']
            show_error_chart(result_df, 'Error')
            st.write(result_df[['Price', 'Predicted Price', 'Error']].describe())

            # Encode categorical columns for visualization
            encoded_df, mappings = encode_categorical_columns(result_df.copy(), ['Rooms', 'Floor'])

            # Decode columns to display text values
            decoded_df = decode_categorical_columns(encoded_df.copy(), mappings)

            # Show grouped bar charts for filtered data
            st.header('Filtered Data Visualization')
            show_grouped_bar_chart(decoded_df, 'Rooms', 'Price', 'Predicted Price')
            show_grouped_bar_chart(decoded_df, 'Floor', 'Price', 'Predicted Price')
    else:
        st.write('Type in the query parameters and click Submit to see the results.')

with tab2:
    entire_df = query_model(df, models[model_input], Location='', Rooms='Wszystkie opcje', Floor='Wszystkie opcje', Area_range=area_range_input)
    # Encode categorical columns for visualization
    encoded_df, mappings = encode_categorical_columns(entire_df.copy(), ['Rooms', 'Floor'])

    # Decode columns to display text values
    decoded_df = decode_categorical_columns(encoded_df.copy(), mappings)

    # Show charts for entire dataset
    st.header('Data Visualization for Entire Dataset')
    show_bar_chart(decoded_df, 'Floor')
    show_bar_chart(decoded_df, 'Rooms')

    # Grouped Bar Charts for entire dataset
    st.header('Grouped Data Visualization for Entire Dataset')
    show_grouped_bar_chart(decoded_df, 'Rooms', 'Price', 'Predicted Price')
    show_grouped_bar_chart(decoded_df, 'Floor', 'Price', 'Predicted Price')

    # Group by state and city for entire dataset
    st.header('Location-based Grouping for Entire Dataset')
    entire_df['State'] = entire_df['Location'].apply(lambda x: x.split(',')[-1].strip())
    entire_df = entire_df[entire_df['State'].str.islower()]
    show_grouped_bar_chart(entire_df, 'State', 'Price', 'Predicted Price')

with tab3:
    st.header('Model Comparison')

    comparison_df = pd.DataFrame(metrics_scores).T
    st.dataframe(comparison_df)

    st.write("Detailed Analysis for Each Model:")
    
    for model_name, model_pipeline in models.items():
        st.subheader(f'Model: {model_name}')
        
        # Get predictions for the entire dataset
        entire_result_df = query_model(df, model_pipeline, Location='', Rooms='Wszystkie opcje', Floor='Wszystkie opcje', Area_range=(float(min_area), float(max_area)))
        
        if entire_result_df is not None:
            # Compute errors
            entire_result_df['Error'] = entire_result_df['Price'] - entire_result_df['Predicted Price']
            
            # Show error chart
            st.write(f"Error Analysis for {model_name}:")
            show_error_chart(entire_result_df, 'Error')
            
            # Display descriptive statistics for predictions
            st.write(entire_result_df[['Price', 'Predicted Price', 'Error']].describe())

