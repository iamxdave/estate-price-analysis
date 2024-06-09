import streamlit as st
import matplotlib.pyplot as plt

# Histogram for a Area
def show_histogram(df, column):
    st.subheader(f"Histogram for {column}")
    fig, ax = plt.subplots()
    ax.hist(df[column], bins=10, color='skyblue', edgecolor='black')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Histogram of {column}')
    st.pyplot(fig)

# Scatter plot for Price vs Area
def show_scatterplot(df, x, y):
    st.subheader(f"Scatter plot for {x} vs {y}")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[x], df[y], color='blue')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f'Scatter plot of {x} vs {y}')
    st.pyplot(fig)

# Bar chart for Rooms
def show_bar_chart(df, column):
    st.subheader(f"Bar chart for {column}")
    fig, ax = plt.subplots()
    column_counts = df[column].value_counts()
    ax.bar(column_counts.index, column_counts.values, color='lightgreen')
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.set_title(f'Bar chart of {column}')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

def show_grouped_bar_chart(df, groupby_column, actual_col, predicted_col=None):
    # Ensure columns exist in DataFrame
    if actual_col not in df.columns:
        st.error(f"Column {actual_col} not found in DataFrame.")
        return
    if predicted_col and predicted_col not in df.columns:
        st.error(f"Column {predicted_col} not found in DataFrame.")
        return

    # Group by the specified column and calculate mean of numeric columns
    grouped_df = df.groupby(groupby_column).mean(numeric_only=True)

    # Ensure the required columns are in the grouped DataFrame
    if predicted_col:
        grouped_df = grouped_df[[actual_col, predicted_col]].dropna()
    else:
        grouped_df = grouped_df[[actual_col]].dropna()

    fig, ax = plt.subplots()
    grouped_df.plot(kind='bar', ax=ax)
    title = f'Average {actual_col}' + (f' and {predicted_col}' if predicted_col else '') + f' by {groupby_column}'
    ax.set_title(title)
    ax.set_ylabel('Price')
    st.pyplot(fig)

@st.cache_data
def show_error_chart(df, error_col):
    fig, ax = plt.subplots()
    df[error_col].plot(kind='hist', bins=30, ax=ax)
    ax.set_title('Prediction Error Distribution')
    ax.set_xlabel('Error')
    st.pyplot(fig)