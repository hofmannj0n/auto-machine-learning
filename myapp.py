import streamlit as st
import json
import os
import yfinance as yf
import pandas as pd
import plotly.express as px

from datetime import datetime
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from pycaret.regression import *


# credentials
API_KEY = "79ab295c19394593829bfd22215cff98"
ENDPOINT = "https://feather-form-recognizer.cognitiveservices.azure.com/"

# global function to create datasets based on user inputs 
def create_dataset(stock, start_date, end_date):
    stock_list = [stock]
    data = yf.download(tickers=stock_list, start=start_date, end=end_date)
    data = data.drop('Adj Close', axis=1)
    data['Ticker'] = stock
    data = data.dropna()
    return data

# Cache the conversion to prevent computation on every rerun
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

# extract text from document 
def extract_text_from_document(document_path):
    document_analysis = DocumentAnalysisClient(endpoint=ENDPOINT, credential=AzureKeyCredential(API_KEY))

    with open(document_path, 'rb') as f:
        poller = document_analysis.begin_analyze_document("prebuilt-document", f.read())
        result = poller.result()

        extracted_text = " "

        for page in result.pages:
            for line in page.lines:
                extracted_text += line.content + " "

        return extracted_text.strip()

# Streamlit app
with st.sidebar:
    st.image('logo.png')
    choice = st.radio('Select one:', [
        'Welcome!  🎊 ', 
        'Data Selection 🔘', 
        'Data Profiling 🔎',
        'Data Visualization 📈', 
        'ML Model Selection 🦾', 
        'Download Your Data! ⬇️' 
    ])

if choice == 'Welcome!  🎊 ':
    st.title("🤖 Automated Machine Learning ")
    st.divider()
    st.subheader("This app was built by Jon Hofmann --> [github](https://github.com/hofmannj0n) 💡", divider='violet')
    st.info("Create your own stock market data, test various ML algorithms to find the best fit for your data, and save your predicted results for future use.")
    st.write("#")

    st.markdown(":rainbow[Tutorial:]")
    st.write("1. Navigate to Data Selection in the sidebar, follow prompts to generate your data for analysis.")
    st.info("You will be prompted to choose your preferred data input method: selecting data directly from the application or by fiilling out a paper order form.")
    st.divider()
    st.write("2. Navigate to Data Profiling in the sidebar")
    st.info("View a profile report of your data, including correlations / charts / observations")
    st.divider()
    st.write("3. Navigate to Data visualizations in the sidebar")
    st.info("View a visualization of the columns in your dataset")
    st.divider()
    st.write("4. Navigate to ML Model Selection and Profiling in the sidebar")
    st.info("Select a target column (column to predict), and view a report of the ML Model scores")
    st.divider()
    st.write("5. Navigate to Download your Data! in the sidebar")
    st.info("Download your data with labeled predictions for further use!")

if choice == "Data Selection 🔘":

    option = st.selectbox(
        'How Would You Like to Select Your Data?',
        ("Fill Out a Paper Order Form", "Choose Data Online"),
        index=None,
        placeholder='Choose Your Data Selection Method'
    )
    st.write("#")

    if option == "Fill Out a Paper Order Form":
        st.header('Paper Order Form Selection', divider='rainbow')
        st.write("""
                Steps: \n
                1. Download a blank form below using "Download Form" button \n
                2. Print + fill out form by hand \n
                3. Scan completed form using Dropbox / Notes and save as a PDF \n
                4. Upload scanned completed form as PDF to OCR Text Extraction window below \n
                5. Select Generate Data buton \n

                Optionally:
                Select "Download Filled Out Form" button and omit steps 2 & 3!
                 """)
        st.divider()
        
        with open("blank-form.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()
    
        btn = st.download_button(
            label="Download Blank Form",
            data=PDFbyte,
            file_name="blank-form.pdf",
            mime="pdf_file"
        )

        with open("filled-out-form.pdf", "rb") as f_file:
            PDFbyt = f_file.read()
    
        btn = st.download_button(
            label="Download Filled Out Form",
            data=PDFbyt,
            file_name="filled-out-form.pdf",
            mime="f_file"
        )

        st.divider()

        st.title("OCR Text Extraction")
        
        uploaded_file = st.file_uploader("Choose a PDF file...", type=["pdf"])

        if uploaded_file is not None:
            
            # Display the uploaded PDF file
            st.write("Uploaded PDF file:", uploaded_file.name)

            # Perform OCR using Azure Form Recognizer
            st.write("Running OCR...")

            # Save the uploaded file temporarily
            temp_file_path = "/tmp/uploaded_pdf.pdf"
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(uploaded_file.read())

            # Extract text 
            extracted_text = extract_text_from_document(temp_file_path)

            # Remove the temporary file
            os.remove(temp_file_path)

            # Format text for Streamlit output
            formatted = extracted_text.split()

            # pop off first 4 items (Form name -> Form Header)
            formatted.pop(0)
            formatted.pop(0)
            formatted.pop(0)
            formatted.pop(0)

            # Seperate the questions / answers into blocks
            block1 = formatted[0:4]
            question1 = ' '.join(block1)
            block2 = formatted[4:6]
            answer1 = ' '.join(block2)
            block3 = formatted[6:22]
            question2 = ' '.join(block3)
            answer2 = formatted[22]
            block5 = formatted[23:27]
            question3 = ' '.join(block5)
            answer3 = formatted[27]
            block6 = formatted[28:32]
            question4 = ' '.join(block6)
            answer4 = formatted[32]

            # Turn question / answer slices into a dict
            formatted_text = {
                question1 : answer1,
                question2 : answer2,
                question3 : answer3,
                question4 : answer4
            }

            # Display the extracted text
            st.divider()
            st.subheader("Extracted Text:")
            st.write(f"{question1} \n {answer1}")
            st.write(f"{question2} \n {answer2}")
            st.write(f"{question3} \n {answer3}")
            st.write(f"{question4} \n {answer4}")
            st.divider()

            # Get the answers needed for model input : Stock, start_date, end_date
            input_answers = []

            for value in formatted_text.values():
                input_answers.append(value)

            input_answers = input_answers[1:4]

            stock = input_answers[0]
            start_date = input_answers[1]
            end_date = input_answers[2]

            # Initialize session state for start / end dates
            if "start_date" not in st.session_state:
                st.session_state.start_date = start_date

            if "end_date" not in st.session_state:
                st.session_state.end_date = end_date


            # Create a generate data button
            if st.button("Generate Data"):

                # Create the dataset based on user input
                data = create_dataset(stock, st.session_state.start_date, st.session_state.end_date)

                # Save the generated data in st.session_state
                st.session_state.data = data
                st.success("Data generated and Ready for Analysis!  Navigate to Data Profiling in SideBar")

                # Display the dataset
                st.write("#")
                st.write("Data Preview:")
                
            if "data" in st.session_state:
                st.write(st.session_state.data)

    elif option == "Choose Data Online":

        # Choice parameters
        stocks = ["AAPL", "GOOGL", "IBM", "MSFT", "VIX", "VOO", "QQQ", "TSLA", "JPM", "AMZN"]
        min_timeframe = datetime.fromisoformat("2015-01-01")
        max_timeframe = datetime.fromisoformat("2023-11-21")

        # Use widgets to get user input
        st.header('Online Data Selection', divider='rainbow')
        st.write("Use the Drop Down Menu and Slider Options to Choose Data")
        st.write("Once Satisfied With Data - Select Generate Data Button")
        st.write("#")
        stock = st.selectbox("Select a Security for Analysis", stocks, key="stock")
        st.write("#")
        start_date = st.slider("Start date", min_value=min_timeframe, max_value=max_timeframe, key="start_date")
        st.write("#")
        end_date = st.slider("End date", min_value=start_date, max_value=max_timeframe, key="end_date")
        st.write("#")

        if st.button("Generate Data"):

            # Create the dataset based on user input
            data = create_dataset(stock, st.session_state.start_date, st.session_state.end_date)

            # Save the generated data in st.session_state
            st.session_state.data = data
            st.success("Data generated and Ready for Analysis!  Navigate to Data Profiling in SideBar")

            # Display the dataset
            st.write("#")
            st.write("Data Preview:")

        # Create the dataset based on user input
        data = create_dataset(st.session_state.stock, st.session_state.start_date, st.session_state.end_date)

        # Display the saved data if it exists
        if "data" in st.session_state:
            st.write(st.session_state.data)

if choice == 'Data Profiling 🔎':
    # Check if data is available in session state
    if "data" in st.session_state:
        st.header('Profile of Your Generated Data!', divider='rainbow')
        profile = ProfileReport(st.session_state.data, title="Profiling Report")
        st_profile_report(profile)
    else:
        st.warning("No data generated yet. Please select 'Data Selection' and generate data first.")

if choice == 'Data Visualization 📈':
        # Check if data is available in session state
    if "data" in st.session_state:

        st.header('Data Visualizations', divider='rainbow')

        df = st.session_state.data
        numeric_columns = df.select_dtypes(include='number').columns
        numeric_columns = numeric_columns[::-1]

        # Create a figure for each numeric column
        for column in numeric_columns:
            fig = px.line(
                df,
                y=column,
                title=f"{column} Line Chart",
                color_discrete_sequence=["#9EE6CF"],
            )
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    else:
        st.warning("No data generated yet. Please select 'Data Selection' and generate data first.")

if choice == 'ML Model Selection 🦾':
    if "data" in st.session_state:
        data = st.session_state.data

        chosen_target = st.selectbox('Choose the Target Column', data.columns)

        if st.button('Run Modelling'): 
            setup(data, target=chosen_target)
            best_model = compare_models()
            
            # Display only a summary of the model comparisons
            st.write("Model Accuracy: (Ranked Most Accurate - to Least Accurate)")
            compare_df = pull()
            st.dataframe(compare_df.head())  # Display only the top rows
            
            # Display only a summary of model predictions
            st.write("Model Predictions: (Using Most Accurate Model)")
            most_accurate_model = predict_model(best_model)
            st.session_state.most_accurate_model = most_accurate_model
            st.dataframe(most_accurate_model.head())  # Display only the top rows

            # Clear unnecessary variables
            del data

    else:
        st.warning("No data generated yet. Please select 'Data Selection' and generate data first.")

if choice == 'Download Your Data! ⬇️':
    # Check if most_accurate_model exists in st.session_state
    if "most_accurate_model" in st.session_state:

        # Access most_accurate_model from st.session_state
        most_accurate_model = st.session_state.most_accurate_model

        # Convert most_accurate_model to CSV
        csv = convert_df(most_accurate_model)

        # Create a download button
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='model_predictions.csv',
            mime='text/csv',
        )

        st.markdown(":blue[Next Steps:]")
        st.write("Check out the documentation for [scikit-learn](https://scikit-learn.org/stable/) and [PyCaret](https://pycaret.gitbook.io/docs/) regarding your most accurate model! ")

        st.code("""
        import pandas as pd
        from pycaret.regression import *
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        # Load the dataset you generated using this app
        df_actual = pd.read_csv("model_predictions.csv)
        
        # Check for accuracy, make future predictions, and so on.  Happy Coding! """, language ="python") 
    else:
        st.warning("No model predictions yet, please navigate to ML Model Selection + Profiling")

