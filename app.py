import streamlit as st
import pandas as pd
import pickle
import google.generativeai as genai
import os

# Configure the Generative AI API
genai.configure(api_key="AIzaSyDyNa6r6LongmI-wyTYKiLcfeukzShd18w")

def get_gemini_response(prompt):
    try:
        model_gen = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model_gen.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Load model artifacts from the 'model' folder
with open('model/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/onehot_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('model/training_columns.pkl', 'rb') as f:
    training_columns = pickle.load(f)

# Title and subtitle of the app
st.markdown("<h1 style='text-align: left;'>🎓 Student Attrition System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: left;'>Predict student dropout using demographic and academic data</h3>", unsafe_allow_html=True)

# File uploader for CSV data
uploaded_file = st.file_uploader("Upload CSV File with Student Data", type=["csv"])

if uploaded_file is not None:
    # Read CSV into a DataFrame
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", data.head())

    # Ensure the CSV contains a 'student_name' column
    if 'student_name' not in data.columns:
        st.error("The CSV file must contain a 'student_name' column.")
    else:
        # Prepare the data for prediction
        # Assume the CSV contains all necessary feature columns with the same names as below
        input_df = data.drop(columns=['student_name'])
        
        # Process categorical features
        categorical_cols = [
            'Application_mode', 'Course', 'Marital_status', 'Nacionality',
            'Mothers_qualification', 'Fathers_qualification',
            'Mothers_occupation', 'Fathers_occupation'
        ]
        try:
            input_encoded = encoder.transform(input_df[categorical_cols])
            input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_cols))
        except Exception as e:
            st.error(f"Error encoding categorical columns: {e}")
        
        input_df = input_df.drop(columns=categorical_cols)
        input_df = pd.concat([input_df.reset_index(drop=True), input_encoded_df.reset_index(drop=True)], axis=1)
        
        # Process numerical features
        numerical_cols = [
            'Previous_qualification_grade', 'Admission_grade',
            'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
            'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
            'Age_at_enrollment'
        ]
        try:
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        except Exception as e:
            st.error(f"Error scaling numerical columns: {e}")
        
        # Align columns to match the training data
        try:
            input_df = input_df[training_columns]
        except Exception as e:
            st.error(f"Error aligning input columns: {e}")

        # Make dropout predictions for all rows
        predictions = model.predict(input_df)
        status_map = {0: 'Dropout', 1: 'Not Dropout'}

        # Create results DataFrame with student names and predictions
        results = data[['student_name']].copy()
        results['Dropout'] = [status_map[p] for p in predictions]
        
        # Generate explanation (reason) for each prediction using Gemini
        reasons = []
        for i, row in data.iterrows():
            prompt = (
                f"Explain the factors that contributed to the dropout prediction for student '{row['student_name']}' "
                f"with the following data: {row.to_dict()} in 10 lines."
            )
            with st.spinner(f"Generating explanation for {row['student_name']}..."):
                explanation = get_gemini_response(prompt)
            reasons.append(explanation)
        
        results['Reason'] = reasons

        st.write("### Prediction Results", results)

else:
    # If no file is uploaded, continue with the interactive input approach
    def user_input_features():
        st.header("Student Demographics")
        col1, col2 = st.columns(2)
        with col1:
            marital_status = st.selectbox(
                'Marital Status',
                [
                    '1 – Single', '2 – Married', '3 – Widower', '4 – Divorced',
                    '5 – Facto Union', '6 – Legally Separated'
                ]
            )
            nationality = st.selectbox(
                'Nacionality',
                [
                    '1 - Portuguese', '2 - German', '6 - Spanish', '11 - Italian', '13 - Dutch', '14 - English',
                    '17 - Lithuanian', '21 - Angolan', '22 - Cape Verdean', '24 - Guinean', '25 - Mozambican',
                    '26 - Santomean', '32 - Turkish', '41 - Brazilian', '62 - Romanian', '100 - Moldova (Republic of)',
                    '101 - Mexican', '103 - Ukrainian', '105 - Russian', '108 - Cuban', '109 - Colombian'
                ]
            )
            gender = st.selectbox('Gender', ['1 – Male', '0 – Female'])
        with col2:
            age_at_enrollment = st.slider('Age at Enrollment', 17, 70, 18)
            displaced = st.selectbox('Displaced', ['1 – Yes', '0 – No'])
            international = st.selectbox('International', ['1 – Yes', '0 – No'])

        st.header("Family Background")
        col3, col4 = st.columns(2)
        with col3:
            mothers_qualification = st.selectbox(
                'Mother\'s Qualification',
                [
                    '1 - Secondary Education - 12th Year of Schooling or Eq.', '2 - Higher Education - Bachelor\'s Degree',
                    '3 - Higher Education - Degree', '4 - Higher Education - Master\'s', '5 - Higher Education - Doctorate',
                    '6 - Frequency of Higher Education', '9 - 12th Year of Schooling - Not Completed',
                    '10 - 11th Year of Schooling - Not Completed', '11 - 7th Year (Old)', '12 - Other - 11th Year of Schooling',
                    '14 - 10th Year of Schooling', '18 - General commerce course', '19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.',
                    '22 - Technical-professional course', '26 - 7th year of schooling', '27 - 2nd cycle of the general high school course',
                    '29 - 9th Year of Schooling - Not Completed', '30 - 8th year of schooling', '34 - Unknown',
                    '35 - Can\'t read or write', '36 - Can read without having a 4th year of schooling', '37 - Basic education 1st cycle (4th/5th year) or equiv.',
                    '38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.', '39 - Technological specialization course',
                    '40 - Higher education - degree (1st cycle)', '41 - Specialized higher studies course',
                    '42 - Professional higher technical course', '43 - Higher Education - Master (2nd cycle)',
                    '44 - Higher Education - Doctorate (3rd cycle)'
                ]
            )
            fathers_qualification = st.selectbox(
                'Father\'s Qualification',
                [
                    '1 - Secondary Education - 12th Year of Schooling or Eq.', '2 - Higher Education - Bachelor\'s Degree',
                    '3 - Higher Education - Degree', '4 - Higher Education - Master\'s', '5 - Higher Education - Doctorate',
                    '6 - Frequency of Higher Education', '9 - 12th Year of Schooling - Not Completed',
                    '10 - 11th Year of Schooling - Not Completed', '11 - 7th Year (Old)', '12 - Other - 11th Year of Schooling',
                    '13 - 2nd year complementary high school course', '14 - 10th Year of Schooling',
                    '18 - General commerce course', '19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.',
                    '20 - Complementary High School Course', '22 - Technical-professional course',
                    '25 - Complementary High School Course - not concluded', '26 - 7th year of schooling',
                    '27 - 2nd cycle of the general high school course', '29 - 9th Year of Schooling - Not Completed',
                    '30 - 8th year of schooling', '31 - General Course of Administration and Commerce',
                    '33 - Supplementary Accounting and Administration', '34 - Unknown', '35 - Can\'t read or write',
                    '36 - Can read without having a 4th year of schooling', '37 - Basic education 1st cycle (4th/5th year) or equiv.',
                    '38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.', '39 - Technological specialization course',
                    '40 - Higher education - degree (1st cycle)', '41 - Specialized higher studies course',
                    '42 - Professional higher technical course', '43 - Higher Education - Master (2nd cycle)',
                    '44 - Higher Education - Doctorate (3rd cycle)'
                ]
            )
        with col4:
            mothers_occupation = st.selectbox(
                'Mother\'s Occupation',
                [
                    '0 - Student', '1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers',
                    '2 - Specialists in Intellectual and Scientific Activities', '3 - Intermediate Level Technicians and Professions',
                    '4 - Administrative staff', '5 - Personal Services, Security and Safety Workers and Sellers',
                    '6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry',
                    '7 - Skilled Workers in Industry, Construction and Craftsmen', '8 - Installation and Machine Operators and Assembly Workers',
                    '9 - Unskilled Workers', '10 - Armed Forces Professions', '90 - Other Situation', '99 - (blank)',
                    '122 - Health professionals', '123 - Teachers', '125 - Specialists in ICT',
                    '131 - Intermediate level science and engineering technicians and professions', '132 - Health technicians',
                    '134 - Legal, social, sports, cultural and similar services technicians',
                    '141 - Office workers and data processing operators', '143 - Data, accounting, statistical and financial operators',
                    '144 - Other administrative support staff', '151 - Personal service workers', '152 - Sellers', '153 - Personal care workers'
                ]
            )
            fathers_occupation = st.selectbox(
                'Father\'s Occupation',
                [
                    '0 - Student', '1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers',
                    '2 - Specialists in Intellectual and Scientific Activities', '3 - Intermediate Level Technicians and Professions',
                    '4 - Administrative staff', '5 - Personal Services, Security and Safety Workers and Sellers',
                    '6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry',
                    '7 - Skilled Workers in Industry, Construction and Craftsmen', '8 - Installation and Machine Operators and Assembly Workers',
                    '9 - Unskilled Workers', '10 - Armed Forces Professions', '90 - Other Situation', '99 - (blank)',
                    '101 - Armed Forces Officers', '102 - Armed Forces Sergeants', '103 - Other Armed Forces personnel',
                    '112 - Administrative and commercial service directors', '114 - Hotel, catering and trade directors',
                    '121 - Physical sciences, mathematics, engineering specialists',
                    '122 - Health professionals', '123 - Teachers', '124 - Finance, accounting and administrative specialists',
                    '131 - Intermediate level science and engineering technicians', '132 - Health technicians',
                    '134 - Legal, social, sports, cultural technicians', '135 - ICT technicians',
                    '141 - Office workers and data processing operators', '143 - Data, accounting, statistical and financial operators',
                    '144 - Other administrative support staff', '151 - Personal service workers', '152 - Sellers', '153 - Personal care workers'
                ]
            )

        st.header("Academic Background")
        col5, col6 = st.columns(2)
        with col5:
            previous_qualification = st.selectbox(
                'Previous Qualification',
                [
                    '1 - Secondary education', '2 - Higher education - bachelor\'s degree', '3 - Higher education - degree',
                    '4 - Higher education - master\'s', '5 - Higher education - doctorate', '6 - Frequency of higher education',
                    '9 - 12th year of schooling - not completed', '10 - 11th year of schooling - not completed',
                    '12 - Other - 11th year of schooling', '14 - 10th year of schooling',
                    '15 - 10th year of schooling - not completed', '19 - Basic education 3rd cycle or equiv.',
                    '38 - Basic education 2nd cycle or equiv.', '39 - Technological specialization course',
                    '40 - Higher education - degree (1st cycle)', '42 - Professional higher technical course',
                    '43 - Higher education - master (2nd cycle)'
                ]
            )
            previous_qualification_grade = st.slider('Previous Qualification Grade', 0.0, 200.0, 150.0)
            admission_grade = st.slider('Admission Grade', 0.0, 200.0, 150.0)
        with col6:
            application_mode = st.selectbox(
                'Application Mode',
                [
                    '1 - 1st phase - general contingent', '2 - Ordinance No. 612/93',
                    '5 - 1st phase - special contingent (Azores Island)', '7 - Holders of other higher courses',
                    '10 - Ordinance No. 854-B/99', '15 - International student (bachelor)',
                    '16 - 1st phase - special contingent (Madeira Island)', '17 - 2nd phase - general contingent',
                    '18 - 3rd phase - general contingent', '26 - Ordinance No. 533-A/99, item b2)',
                    '27 - Ordinance No. 533-A/99, item b3 (Other Institution)', '39 - Over 23 years old',
                    '42 - Transfer', '43 - Change of course', '44 - Technological specialization diploma holders',
                    '51 - Change of institution/course', '53 - Short cycle diploma holders',
                    '57 - Change of institution/course (International)'
                ]
            )
            application_order = st.slider('Application Order', 0, 9, 0)
            course = st.selectbox(
                'Course',
                [
                    '33 - Biofuel Production Technologies', '171 - Animation and Multimedia Design',
                    '8014 - Social Service (evening attendance)', '9003 - Agronomy',
                    '9070 - Communication Design', '9085 - Veterinary Nursing', '9119 - Informatics Engineering',
                    '9130 - Equinculture', '9147 - Management', '9238 - Social Service', '9254 - Tourism',
                    '9500 - Nursing', '9556 - Oral Hygiene', '9670 - Advertising and Marketing Management',
                    '9773 - Journalism and Communication', '9853 - Basic Education', '9991 - Management (evening attendance)'
                ]
            )

        st.header("Current Academic Performance")
        col7, col8 = st.columns(2)
        with col7:
            daytime_evening_attendance = st.selectbox('Daytime/Evening Attendance', ['1 – Daytime', '0 - Evening'])
            curricular_units_1st_sem_credited = st.slider('Curricular Units 1st Sem (Credited)', 0, 60, 30)
            curricular_units_1st_sem_enrolled = st.slider('Curricular Units 1st Sem (Enrolled)', 0, 60, 30)
        with col8:
            curricular_units_1st_sem_evaluations = st.slider('Curricular Units 1st Sem (Evaluations)', 0, 60, 30)
            curricular_units_1st_sem_approved = st.slider('Curricular Units 1st Sem (Approved)', 0, 60, 30)

        st.header("Additional Information")
        col9, col10 = st.columns(2)
        with col9:
            educational_special_needs = st.selectbox('Educational Special Needs', ['1 – Yes', '0 – No'])
            debtor = st.selectbox('Debtor', ['1 – Yes', '0 – No'])
        with col10:
            tuition_fees_up_to_date = st.selectbox('Tuition Fees Up to Date', ['1 – Yes', '0 – No'])
            scholarship_holder = st.selectbox('Scholarship Holder', ['1 – Yes', '0 – No'])

        data = {
            'Marital_status': int(marital_status.split(' – ')[0]),
            'Application_mode': int(application_mode.split(' - ')[0]),
            'Application_order': application_order,
            'Course': int(course.split(' - ')[0]),
            'Daytime_evening_attendance': int(daytime_evening_attendance.split(' – ')[0]),
            'Previous_qualification': int(previous_qualification.split(' - ')[0]),
            'Previous_qualification_grade': previous_qualification_grade,
            'Nacionality': int(nationality.split(' - ')[0]),
            'Mothers_qualification': int(mothers_qualification.split(' - ')[0]),
            'Fathers_qualification': int(fathers_qualification.split(' - ')[0]),
            'Mothers_occupation': int(mothers_occupation.split(' - ')[0]),
            'Fathers_occupation': int(fathers_occupation.split(' - ')[0]),
            'Admission_grade': admission_grade,
            'Displaced': int(displaced.split(' – ')[0]),
            'Educational_special_needs': int(educational_special_needs.split(' – ')[0]),
            'Debtor': int(debtor.split(' – ')[0]),
            'Tuition_fees_up_to_date': int(tuition_fees_up_to_date.split(' – ')[0]),
            'Gender': int(gender.split(' – ')[0]),
            'Scholarship_holder': int(scholarship_holder.split(' – ')[0]),
            'Age_at_enrollment': age_at_enrollment,
            'International': int(international.split(' – ')[0]),
            'Curricular_units_1st_sem_credited': curricular_units_1st_sem_credited,
            'Curricular_units_1st_sem_enrolled': curricular_units_1st_sem_enrolled,
            'Curricular_units_1st_sem_evaluations': curricular_units_1st_sem_evaluations,
            'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()
    if st.button("Predict Dropout"):
        categorical_cols = [
            'Application_mode', 'Course', 'Marital_status', 'Nacionality',
            'Mothers_qualification', 'Fathers_qualification',
            'Mothers_occupation', 'Fathers_occupation'
        ]
        input_encoded = encoder.transform(input_df[categorical_cols])
        input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_cols))
        input_df = input_df.drop(columns=categorical_cols)
        input_df = pd.concat([input_df.reset_index(drop=True), input_encoded_df.reset_index(drop=True)], axis=1)
        numerical_cols = [
            'Previous_qualification_grade', 'Admission_grade',
            'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
            'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
            'Age_at_enrollment'
        ]
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        input_df = input_df[training_columns]
        prediction = model.predict(input_df)
        st.subheader('Prediction')
        st.write(status_map[prediction[0]])
        
        prompt = f"Explain the factors that contributed to the dropout prediction for the student with the following data: {status_map[prediction[0]]} in 10 lines"
        with st.spinner("Thinking..."):
            response = get_gemini_response(prompt)
        st.write("### AI Response:")
        st.write(response)
