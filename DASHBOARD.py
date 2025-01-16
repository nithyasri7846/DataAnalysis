'''
Interactive Dashboard for Data Insights
This project is a Streamlit-based interactive dashboard that allows users to upload datasets, explore their structure, and visualize key insights through various charts and graphs.

Features
Dataset Overview: Displays the dataset's first few rows, shape, column names, data types, and descriptive statistics.
Correlation Heatmap: Visualizes the correlation between numeric columns in the dataset.
Bar Chart: Allows users to create a bar chart by selecting numeric columns for the X and Y axes.
Line Chart: Enables users to generate a line chart by selecting numeric columns for the X and Y axes.
Interactive Interface: User-friendly layout with a sidebar for easy navigation and dataset upload.
'''

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the page title and layout
st.set_page_config(page_title="Interactive Dashboard", layout="wide")

# Header
st.title("Interactive Dashboard for Data Insights")
st.markdown("This dashboard provides insights into the uploaded dataset, including visualizations and statistical analysis.")

# Sidebar for uploading the dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
    
    # Display the dataset
    st.header("Dataset Overview")
    st.write("Here is a preview of the dataset:")
    st.write(data.head())

    # Basic Dataset Info
    st.subheader("Basic Information")
    st.write("Shape of the dataset:", data.shape)
    st.write("Column names:", data.columns.tolist())
    st.write("Data types:")
    st.write(data.dtypes)
    
    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.write(data.describe())

    # Visualization Section
    st.header("Visualizations")

    # Correlation Heatmap (Filter only numeric columns)
    st.subheader("Correlation Heatmap")
    numeric_data = data.select_dtypes(include=["number"])  # Filter only numeric columns
    if not numeric_data.empty:  # Ensure there are numeric columns
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numeric columns available to generate a heatmap.")

    # Select columns for plotting
    st.subheader("Bar Chart and Line Chart")
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns.tolist()

    if len(numeric_columns) > 1:
        col1, col2 = st.columns(2)

        # Bar Chart
        with col1:
            st.write("Bar Chart")
            x_axis = st.selectbox("Select X-axis for Bar Chart", options=numeric_columns, key="bar_x")
            y_axis = st.selectbox("Select Y-axis for Bar Chart", options=numeric_columns, key="bar_y")
            if st.button("Generate Bar Chart"):
                fig, ax = plt.subplots()
                ax.bar(data[x_axis], data[y_axis], color="skyblue")
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f"Bar Chart: {x_axis} vs {y_axis}")
                st.pyplot(fig)

        # Line Chart
        with col2:
            st.write("Line Chart")
            line_x = st.selectbox("Select X-axis for Line Chart", options=numeric_columns, key="line_x")
            line_y = st.selectbox("Select Y-axis for Line Chart", options=numeric_columns, key="line_y")
            if st.button("Generate Line Chart"):
                fig, ax = plt.subplots()
                ax.plot(data[line_x], data[line_y], marker="o", color="green")
                ax.set_xlabel(line_x)
                ax.set_ylabel(line_y)
                ax.set_title(f"Line Chart: {line_x} vs {line_y}")
                st.pyplot(fig)
    else:
        st.warning("The dataset must have at least two numeric columns for visualization.")

else:
    st.info("Please upload a CSV file to get started.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed with ❤️ by NS")
