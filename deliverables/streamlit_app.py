import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import numpy as np
from sklearn.pipeline import Pipeline
import matplotlib as plt

# Import custom_transformer methods for our pipelines. We do this to keep the notebook clean and easy to read and to be able to adapt code quickly and easily
sys.path.append('../pipeline_scripts')
from incoherences_custom_transformers import *
from preprocessing_pipeline import *
from missing_values_transformers import *

# Assume data is already loaded into a dataframe called `data`
# data = pd.read_csv("../project_data/train_data.csv", dtype={'Zip Code': str})  # for example

#model = joblib.load('model.pkl')

st.title("Prediction Interface")
st.write("Please select values for each feature to get an instant prediction.")

# Define the columns and data type info
personal_info_cols = ['Gender', 'Number of Dependents']

incident_details_cols = ['Age at Injury', 'Accident Date', 'Assembly Date', 'C-2 Date', 'C-3 Date', 'First Hearing Date', 
                         'IME-4 Count', 'COVID-19 Indicator', 'County of Injury', 'WCIO Cause of Injury Description', 
                         'WCIO Nature of Injury Description', 'WCIO Part Of Body Description', 'Medical Fee Region', 
                         'District Name', 'Alternative Dispute Resolution']

industry_employment_cols = ['Industry Code Description', 'Carrier Type', 'Attorney/Representative', 'Average Weekly Wage']


# Identify date columns (currently stored as object but represent dates)
date_columns = [
    "Accident Date",
    "Assembly Date",
    "C-2 Date",
    "C-3 Date",
    "First Hearing Date"
]

# # Convert these columns to datetime
# for dcol in date_columns:
#     data[dcol] = pd.to_datetime(data[dcol], errors='coerce')

# Define the columns and their types
columns_info = [
    ("Accident Date", "object"),  # date column
    ("Age at Injury", "float64"), # will be handled as int
    ("Alternative Dispute Resolution", "object"),
    ("Assembly Date", "object"),  # date column
    ("Attorney/Representative", "object"),
    ("Average Weekly Wage", "float64"),
    ("C-2 Date", "object"),       # date column
    ("C-3 Date", "object"),       # date column
    ("Carrier Name", "object"),
    ("Carrier Type", "object"),
    # ("Claim Injury Type", "object"),
    ("County of Injury", "object"),
    ("COVID-19 Indicator", "object"),
    # ("District Name", "object"),
    ("First Hearing Date", "object"),  # date column
    ("Gender", "object"),
    ("IME-4 Count", "float64"),   # will be handled as int
    ("Industry Code Description", "object"),
    ("Medical Fee Region", "object"),
    ("WCIO Cause of Injury Description", "object"),
    ("WCIO Nature of Injury Description", "object"),
    ("WCIO Part Of Body Description", "object"),
    # ("Zip Code", "object"),
    # ("WCB Decision", "object"),
    ("Number of Dependents", "float64")
]

unique_value_dict = {'Alternative Dispute Resolution': ['N', 'Y', 'U'],
 'Attorney/Representative': ['N', 'Y'],
 'Carrier Type': ['1A. PRIVATE', '2A. SIF', '4A. SELF PRIVATE', '3A. SELF PUBLIC', 'UNKNOWN', '5D. SPECIAL FUND - UNKNOWN', '5A. SPECIAL FUND - CONS. COMM. (SECT. 25-A)', '5C. SPECIAL FUND - POI CARRIER WCB MENANDS'],
 'County of Injury': ['ST. LAWRENCE', 'WYOMING', 'ORANGE', 'DUTCHESS', 'SUFFOLK', 'ONONDAGA', 'RICHMOND', 'MONROE', 'KINGS', 'NEW YORK', 'QUEENS', 'WESTCHESTER', 'GREENE', 'NASSAU', 'ALBANY', 'ERIE', 'BRONX', 'CAYUGA', 'NIAGARA', 'LIVINGSTON', 'WASHINGTON', 'MADISON', 'WARREN', 'SENECA', 'GENESEE', 'SARATOGA', 'CHAUTAUQUA', 'COLUMBIA', 'RENSSELAER', 'CATTARAUGUS', 'ROCKLAND', 'SCHUYLER', 'BROOME', 'ULSTER', 'CLINTON', 'ONEIDA', 'UNKNOWN', 'MONTGOMERY', 'ONTARIO', 'SCHENECTADY', 'CHEMUNG', 'YATES', 'HERKIMER', 'ALLEGANY', 'TIOGA', 'FULTON', 'DELAWARE', 'TOMPKINS', 'PUTNAM', 'OSWEGO', 'LEWIS', 'ESSEX', 'OTSEGO', 'CORTLAND', 'ORLEANS', 'SULLIVAN', 'CHENANGO', 'FRANKLIN', 'WAYNE', 'JEFFERSON', 'STEUBEN', 'SCHOHARIE', 'HAMILTON'],
 'COVID-19 Indicator': ['N', 'Y'],
 'District Name': ['SYRACUSE', 'ROCHESTER', 'ALBANY', 'HAUPPAUGE', 'NYC', 'BUFFALO', 'BINGHAMTON', 'STATEWIDE'],
 'Gender': ['M', 'F', 'U', 'X'],
 'Medical Fee Region': ['I', 'II', 'IV', 'UK', 'III'],
 'Industry Code Description': ['RETAIL TRADE', 'CONSTRUCTION', 'ADMINISTRATIVE AND SUPPORT AND WASTE MANAGEMENT AND REMEDIAT', 'HEALTH CARE AND SOCIAL ASSISTANCE', 'ACCOMMODATION AND FOOD SERVICES', 'EDUCATIONAL SERVICES', 'INFORMATION', 'MANUFACTURING', 'TRANSPORTATION AND WAREHOUSING', 'WHOLESALE TRADE', 'REAL ESTATE AND RENTAL AND LEASING', 'FINANCE AND INSURANCE', 'OTHER SERVICES (EXCEPT PUBLIC ADMINISTRATION)', 'PUBLIC ADMINISTRATION', 'PROFESSIONAL, SCIENTIFIC, AND TECHNICAL SERVICES', 'ARTS, ENTERTAINMENT, AND RECREATION', 'UTILITIES', 'AGRICULTURE, FORESTRY, FISHING AND HUNTING', 'MINING', 'MANAGEMENT OF COMPANIES AND ENTERPRISES'],
 'WCIO Cause of Injury Description': ['FROM LIQUID OR GREASE SPILLS', 'REPETITIVE MOTION', 'OBJECT BEING LIFTED OR HANDLED', 'HAND TOOL, UTENSIL; NOT POWERED', 'FALL, SLIP OR TRIP, NOC', 'CUT, PUNCTURE, SCRAPE, NOC', 'OTHER - MISCELLANEOUS, NOC', 'STRUCK OR INJURED, NOC', 'FALLING OR FLYING OBJECT', 'CHEMICALS', 'COLLISION OR SIDESWIPE WITH ANOTHER VEHICLE', 'LIFTING', 'TWISTING', 'ON SAME LEVEL', 'STRAIN OR INJURY BY, NOC', 'MOTOR VEHICLE, NOC', 'FROM DIFFERENT LEVEL (ELEVATION)', 'PUSHING OR PULLING', 'FOREIGN MATTER (BODY) IN EYE(S)', 'FELLOW WORKER, PATIENT OR OTHER PERSON', 'STEAM OR HOT FLUIDS', 'STATIONARY OBJECT', 'ON ICE OR SNOW', 'ABSORPTION, INGESTION OR INHALATION, NOC', 'PERSON IN ACT OF A CRIME', 'INTO OPENINGS', 'ON STAIRS', 'FROM LADDER OR SCAFFOLDING', 'SLIP, OR TRIP, DID NOT FALL', 'JUMPING OR LEAPING', 'MOTOR VEHICLE', 'RUBBED OR ABRADED, NOC', 'REACHING', 'OBJECT HANDLED', 'HOT OBJECTS OR SUBSTANCES', 'ELECTRICAL CURRENT', 'HOLDING OR CARRYING', 'CAUGHT IN, UNDER OR BETWEEN, NOC', 'FIRE OR FLAME', 'CUMULATIVE, NOC', 'POWERED HAND TOOL, APPLIANCE', 'STRIKING AGAINST OR STEPPING ON, NOC', 'MACHINE OR MACHINERY', 'COLD OBJECTS OR SUBSTANCES', 'BROKEN GLASS', 'COLLISION WITH A FIXED OBJECT', 'STEPPING ON SHARP OBJECT', 'OBJECT HANDLED BY OTHERS', 'DUST, GASES, FUMES OR VAPORS', 'OTHER THAN PHYSICAL CAUSE OF INJURY', 'CONTACT WITH, NOC', 'USING TOOL OR MACHINERY', 'SANDING, SCRAPING, CLEANING OPERATION', 'CONTINUAL NOISE', 'ANIMAL OR INSECT', 'MOVING PARTS OF MACHINE', 'GUNSHOT', 'WIELDING OR THROWING', 'MOVING PART OF MACHINE', 'TEMPERATURE EXTREMES', 'HAND TOOL OR MACHINE IN USE', 'VEHICLE UPSET', 'COLLAPSING MATERIALS (SLIDES OF EARTH)', 'TERRORISM', 'PANDEMIC', 'WELDING OPERATION', 'NATURAL DISASTERS', 'EXPLOSION OR FLARE BACK', 'RADIATION', 'CRASH OF RAIL VEHICLE', 'MOLD', 'ABNORMAL AIR PRESSURE', 'CRASH OF WATER VEHICLE', 'CRASH OF AIRPLANE'],
 'WCIO Nature of Injury Description': ['CONTUSION', 'SPRAIN OR TEAR', 'CONCUSSION', 'PUNCTURE', 'LACERATION', 'ALL OTHER OCCUPATIONAL DISEASE INJURY, NOC', 'ALL OTHER SPECIFIC INJURIES, NOC', 'INFLAMMATION', 'BURN', 'STRAIN OR TEAR', 'FRACTURE', 'FOREIGN BODY', 'MULTIPLE PHYSICAL INJURIES ONLY', 'RUPTURE', 'DISLOCATION', 'ALL OTHER CUMULATIVE INJURY, NOC', 'HERNIA', 'ANGINA PECTORIS', 'CARPAL TUNNEL SYNDROME', 'NO PHYSICAL INJURY', 'INFECTION', 'CRUSHING', 'SYNCOPE', 'POISONING - GENERAL (NOT OD OR CUMULATIVE', 'RESPIRATORY DISORDERS', 'HEARING LOSS OR IMPAIRMENT', 'MENTAL STRESS', 'SEVERANCE', 'ELECTRIC SHOCK', 'LOSS OF HEARING', 'DUST DISEASE, NOC', 'DERMATITIS', 'ASPHYXIATION', 'MENTAL DISORDER', 'CONTAGIOUS DISEASE', 'AMPUTATION', 'MYOCARDIAL INFARCTION', 'POISONING - CHEMICAL, (OTHER THAN METALS)', 'MULTIPLE INJURIES INCLUDING BOTH PHYSICAL AND PSYCHOLOGICAL', 'VISION LOSS', 'VASCULAR', 'COVID-19', 'CANCER', 'HEAT PROSTRATION', 'AIDS', 'ENUCLEATION', 'ASBESTOSIS', 'POISONING - METAL', 'VDT - RELATED DISEASES', 'FREEZING', 'BLACK LUNG', 'SILICOSIS', 'ADVERSE REACTION TO A VACCINATION OR INOCULATION', 'HEPATITIS C', 'RADIATION', 'BYSSINOSIS'],
 'WCIO Part Of Body Description': ['BUTTOCKS', 'SHOULDER(S)', 'MULTIPLE HEAD INJURY', 'FINGER(S)', 'LUNGS', 'EYE(S)', 'ANKLE', 'KNEE', 'THUMB', 'LOWER BACK AREA', 'ABDOMEN INCLUDING GROIN', 'LOWER LEG', 'HIP', 'UPPER LEG', 'MOUTH', 'WRIST', 'SPINAL CORD', 'HAND', 'SOFT TISSUE', 'UPPER ARM', 'FOOT', 'ELBOW', 'MULTIPLE UPPER EXTREMITIES', 'MULTIPLE BODY PARTS (INCLUDING BODY', 'BODY SYSTEMS AND MULTIPLE BODY SYSTEMS', 'MULTIPLE NECK INJURY', 'CHEST', 'WRIST (S) & HAND(S)', 'EAR(S)', 'MULTIPLE LOWER EXTREMITIES', 'DISC', 'LOWER ARM', 'MULTIPLE', 'UPPER BACK AREA', 'SKULL', 'TOES', 'FACIAL BONES', 'TEETH', 'NO PHYSICAL INJURY', 'MULTIPLE TRUNK', 'WHOLE BODY', 'INSUFFICIENT INFO TO PROPERLY IDENTIFY - UNCLASSIFIED', 'PELVIS', 'NOSE', 'GREAT TOE', 'INTERNAL ORGANS', 'HEART', 'VERTEBRAE', 'LUMBAR & OR SACRAL VERTEBRAE (VERTEBRA', 'BRAIN', 'SACRUM AND COCCYX', 'ARTIFICIAL APPLIANCE', 'LARYNX', 'TRACHEA'],
 'Number of Dependents': ['1.0', '4.0', '6.0', '5.0', '3.0', '2.0', '0.0']}


C_o_In_dict = {'ABNORMAL AIR PRESSURE': 14.0, 'ABSORPTION, INGESTION OR INHALATION, NOC': 82.0,
 'ANIMAL OR INSECT': 85.0, 'BROKEN GLASS': 15.0, 'CAUGHT IN, UNDER OR BETWEEN, NOC': 13.0, 'CHEMICALS': 1.0, 'COLD OBJECTS OR SUBSTANCES': 11.0, 'COLLAPSING MATERIALS (SLIDES OF EARTH)': 20.0, 'COLLISION OR SIDESWIPE WITH ANOTHER VEHICLE': 45.0, 'COLLISION WITH A FIXED OBJECT': 46.0, 'CONTACT WITH, NOC': 9.0, 'CONTINUAL NOISE': 52.0, 'CRASH OF AIRPLANE': 47.0,
 'CRASH OF RAIL VEHICLE': 41.0, 'CRASH OF WATER VEHICLE': 40.0, 'CUMULATIVE, NOC': 98.0, 'CUT, PUNCTURE, SCRAPE, NOC': 19.0, 'DUST, GASES, FUMES OR VAPORS': 6.0, 'ELECTRICAL CURRENT': 84.0, 'EXPLOSION OR FLARE BACK': 86.0, 'FALL, SLIP OR TRIP, NOC': 31.0, 'FALLING OR FLYING OBJECT': 75.0, 'FELLOW WORKER, PATIENT OR OTHER PERSON': 74.0, 'FIRE OR FLAME': 4.0, 'FOREIGN MATTER (BODY) IN EYE(S)': 87.0, 'FROM DIFFERENT LEVEL (ELEVATION)': 25.0, 'FROM LADDER OR SCAFFOLDING': 26.0, 'FROM LIQUID OR GREASE SPILLS': 27.0, 'GUNSHOT': 93.0, 'HAND TOOL OR MACHINE IN USE': 76.0, 'HAND TOOL, UTENSIL; NOT POWERED': 16.0, 'HOLDING OR CARRYING': 55.0, 'HOT OBJECTS OR SUBSTANCES': 2.0, 'INTO OPENINGS': 28.0, 'JUMPING OR LEAPING': 54.0, 'LIFTING': 56.0, 'MACHINE OR MACHINERY': 10.0,
 'MOLD': 91.0, 'MOTOR VEHICLE': 77.0, 'MOTOR VEHICLE, NOC': 50.0, 'MOVING PART OF MACHINE': 65.0, 'MOVING PARTS OF MACHINE': 78.0, 'NATURAL DISASTERS': 88.0, 'OBJECT BEING LIFTED OR HANDLED': 79.0, 'OBJECT HANDLED': 12.0, 'OBJECT HANDLED BY OTHERS': 80.0, 'ON ICE OR SNOW': 32.0, 'ON SAME LEVEL': 29.0, 'ON STAIRS': 33.0, 'OTHER - MISCELLANEOUS, NOC': 99.0, 'OTHER THAN PHYSICAL CAUSE OF INJURY': 90.0, 'PANDEMIC': 83.0, 'PERSON IN ACT OF A CRIME': 89.0, 'POWERED HAND TOOL, APPLIANCE': 18.0, 'PUSHING OR PULLING': 57.0, 'RADIATION': 8.0, 'REACHING': 58.0, 'REPETITIVE MOTION': 94.0, 'RUBBED OR ABRADED, NOC': 95.0,
 'SANDING, SCRAPING, CLEANING OPERATION': 67.0, 'SLIP, OR TRIP, DID NOT FALL': 30.0, 'STATIONARY OBJECT': 68.0, 'STEAM OR HOT FLUIDS': 5.0, 'STEPPING ON SHARP OBJECT': 69.0, 'STRAIN OR INJURY BY, NOC': 60.0, 'STRIKING AGAINST OR STEPPING ON, NOC': 70.0, 'STRUCK OR INJURED, NOC': 81.0, 'TEMPERATURE EXTREMES': 3.0, 'TERRORISM': 96.0, 'TWISTING': 53.0, 'USING TOOL OR MACHINERY': 59.0, 'VEHICLE UPSET': 48.0, 'WELDING OPERATION': 7.0, 'WIELDING OR THROWING': 61.0,
 np.nan : np.nan}

N_o_In_dict = {'ADVERSE REACTION TO A VACCINATION OR INOCULATION': 38.0,
'AIDS': 75.0,'ALL OTHER CUMULATIVE INJURY, NOC': 80.0,'ALL OTHER OCCUPATIONAL DISEASE INJURY, NOC': 71.0,'ALL OTHER SPECIFIC INJURIES, NOC': 59.0,'AMPUTATION': 2.0,'ANGINA PECTORIS': 3.0,'ASBESTOSIS': 61.0,'ASPHYXIATION': 54.0,'BLACK LUNG': 62.0,'BURN': 4.0,'BYSSINOSIS': 63.0,'CANCER': 74.0,'CARPAL TUNNEL SYNDROME': 78.0,'CONCUSSION': 7.0,'CONTAGIOUS DISEASE': 73.0,'CONTUSION': 10.0,'COVID-19': 83.0,'CRUSHING': 13.0,'DERMATITIS': 68.0, 'DISLOCATION': 16.0,
 'DUST DISEASE, NOC': 60.0, 'ELECTRIC SHOCK': 19.0, 'ENUCLEATION': 22.0, 'FOREIGN BODY': 25.0, 'FRACTURE': 28.0, 'FREEZING': 30.0, 'HEARING LOSS OR IMPAIRMENT': 31.0, 'HEAT PROSTRATION': 32.0, 'HEPATITIS C': 79.0, 'HERNIA': 34.0, 'INFECTION': 36.0, 'INFLAMMATION': 37.0, 'LACERATION': 40.0, 'LOSS OF HEARING': 72.0,
 'MENTAL DISORDER': 69.0, 'MENTAL STRESS': 77.0, 'MULTIPLE INJURIES INCLUDING BOTH PHYSICAL AND PSYCHOLOGICAL': 91.0, 'MULTIPLE PHYSICAL INJURIES ONLY': 90.0, 'MYOCARDIAL INFARCTION': 41.0, 'NO PHYSICAL INJURY': 1.0, 'POISONING - CHEMICAL, (OTHER THAN METALS)': 66.0, 'POISONING - GENERAL (NOT OD OR CUMULATIVE': 42.0, 'POISONING - METAL': 67.0, 'PUNCTURE': 43.0, 'RADIATION': 70.0, 'RESPIRATORY DISORDERS': 65.0, 'RUPTURE': 46.0, 'SEVERANCE': 47.0, 'SILICOSIS': 64.0, 'SPRAIN OR TEAR': 49.0, 'STRAIN OR TEAR': 52.0, 'SYNCOPE': 53.0, 'VASCULAR': 55.0, 'VDT - RELATED DISEASES': 76.0, 'VISION LOSS': 58.0, 
 np.nan : np.nan}

P_o_B_dict = {'ABDOMEN INCLUDING GROIN': 61.0,
 'ANKLE': 55.0, 'ARTIFICIAL APPLIANCE': 64.0, 'BODY SYSTEMS AND MULTIPLE BODY SYSTEMS': 91.0, 'BRAIN': 12.0, 'BUTTOCKS': 62.0, 'CHEST': 44.0, 'DISC': 22.0, 'EAR(S)': 13.0, 'ELBOW': 32.0, 'EYE(S)': 14.0, 'FACIAL BONES': 19.0, 'FINGER(S)': 36.0, 'FOOT': 56.0, 'GREAT TOE': 58.0, 'HAND': 35.0, 'HEART': 49.0, 'HIP': 51.0, 'INSUFFICIENT INFO TO PROPERLY IDENTIFY - UNCLASSIFIED': 65.0, 'INTERNAL ORGANS': 48.0, 'KNEE': 53.0, 'LARYNX': 24.0, 'LOWER ARM': 33.0, 'LOWER BACK AREA': 42.0, 'LOWER LEG': 54.0, 'LUMBAR & OR SACRAL VERTEBRAE (VERTEBRA': 63.0, 'LUNGS': 60.0, 'MOUTH': 17.0, 'MULTIPLE': -9.0, 'MULTIPLE BODY PARTS (INCLUDING BODY': 90.0, 'MULTIPLE HEAD INJURY': 10.0,
 'MULTIPLE LOWER EXTREMITIES': 50.0, 'MULTIPLE NECK INJURY': 20.0, 'MULTIPLE TRUNK': 40.0, 'MULTIPLE UPPER EXTREMITIES': 30.0, 'NO PHYSICAL INJURY': 66.0, 'NOSE': 15.0, 'PELVIS': 46.0, 'SACRUM AND COCCYX': 45.0, 'SHOULDER(S)': 38.0, 'SKULL': 11.0, 'SOFT TISSUE': 25.0, 'SPINAL CORD': 23.0, 'TEETH': 16.0, 'THUMB': 37.0, 'TOES': 57.0, 'TRACHEA': 26.0, 'UPPER ARM': 31.0, 'UPPER BACK AREA': 41.0, 'UPPER LEG': 52.0, 'VERTEBRAE': 21.0, 'WHOLE BODY': 99.0, 'WRIST': 34.0, 'WRIST (S) & HAND(S)': 39.0,
 np.nan : np.nan}


Ind_dict = {'ACCOMMODATION AND FOOD SERVICES': 72.0,
 'ADMINISTRATIVE AND SUPPORT AND WASTE MANAGEMENT AND REMEDIAT': 56.0, 'AGRICULTURE, FORESTRY, FISHING AND HUNTING': 11.0, 'ARTS, ENTERTAINMENT, AND RECREATION': 71.0, 'CONSTRUCTION': 23.0, 'EDUCATIONAL SERVICES': 61.0, 'FINANCE AND INSURANCE': 52.0, 'HEALTH CARE AND SOCIAL ASSISTANCE': 62.0, 'INFORMATION': 51.0, 'MANAGEMENT OF COMPANIES AND ENTERPRISES': 55.0, 'MANUFACTURING': 33.0, 'MINING': 21.0,
 'OTHER SERVICES (EXCEPT PUBLIC ADMINISTRATION)': 81.0,
 'PROFESSIONAL, SCIENTIFIC, AND TECHNICAL SERVICES': 54.0, 'PUBLIC ADMINISTRATION': 92.0, 'REAL ESTATE AND RENTAL AND LEASING': 53.0, 'RETAIL TRADE': 44.0, 'TRANSPORTATION AND WAREHOUSING': 49.0, 'UTILITIES': 22.0, 'WHOLESALE TRADE': 42.0, 
 np.nan : np.nan}



# Function to handle user inputs based on column type
def handle_column_input(col, dtype, unique_value_dict):
    user_input = None
    
    if col in date_columns:
        # ---------------------------------------
        # Special handling for "Accident Date"
        # ---------------------------------------
        if col == "Accident Date":
            # Show only the text input for the date
            date_input = st.text_input(
                f"{col} format: (YYYY-MM-DD)",
                value=pd.Timestamp.now().strftime("%Y-%m-%d")
            )
            try:
                user_input = pd.to_datetime(date_input).date()
                if user_input > pd.Timestamp.now().date():
                    st.error(f"Invalid date for {col}. Please use a past date.")
                    user_input = np.nan
            except ValueError:
                st.error(f"Invalid date format for {col}. Please use YYYY-MM-DD.")
                user_input = np.nan
        
        else:
            # Default handling for other date columns (with Unknown checkbox)
            unknown_selected = st.checkbox(f"Unknown {col}?", value=False)
            if unknown_selected:
                user_input = np.nan
            else:
                date_input = st.text_input(
                    f"{col} format: (YYYY-MM-DD)",
                    value=pd.Timestamp.now().strftime("%Y-%m-%d")
                )
                try:
                    user_input = pd.to_datetime(date_input).date()
                    if user_input > pd.Timestamp.now().date():
                        st.error(f"Invalid date for {col}. Please use a past date.")
                        user_input = np.nan
                except ValueError:
                    st.error(f"Invalid date format for {col}. Please use YYYY-MM-DD.")
                    user_input = np.nan
    
    elif dtype == "object":
        # Object columns: Selectbox with unique values + "Unknown"
        unique_values = unique_value_dict.get(col, [])
        unique_values = ["Unknown"] + unique_values
        selected = st.selectbox(f"{col}", unique_values)
        user_input = np.nan if selected == "Unknown" else selected

    else:
        # Numeric columns: Handle int and float types
        if col == "Age at Injury" or col == "IME-4 Count":
            default_val = 30 if col == "Age at Injury" else 0
            user_input = int(st.number_input(f"{col}", value=default_val, step=1))
        else:
            default_val = 1
            user_input = float(st.number_input(f"{col}", value=default_val, step=1))

    return user_input



# Set up the tabs for each section
tabs = st.tabs(["Personal Information", "Incident Details", "Industry and Employment"])

# Initialize dictionary to store user inputs
user_inputs = {}

# Process inputs for Personal Informationd
with tabs[0]:  # Personal Information
    for col in personal_info_cols:
        dtype = "object" if col in ['Gender'] else "numeric"  # Set dtype based on the column
        user_inputs[col] = handle_column_input(col, dtype, unique_value_dict)

# Process inputs for Incident Details
with tabs[1]:  # Incident Details
    for col in incident_details_cols:
        dtype = "date" if "Date" in col else "object" if col in unique_value_dict else "numeric"
        user_inputs[col] = handle_column_input(col, dtype, unique_value_dict)

# Process inputs for Industry and Employment
with tabs[2]:  # Industry and Employment
    for col in industry_employment_cols:
        dtype = "object" if col in unique_value_dict else "numeric"
        user_inputs[col] = handle_column_input(col, dtype, unique_value_dict)




# hard coding carrier name to avoid pipeline error
# Carrier Name 
user_inputs['Carrier Name'] = 'placeholder_hardcoded'


# Birth Year
user_inputs['Birth Year'] = user_inputs['Accident Date'].year - user_inputs['Age at Injury']
# set to datetime object
user_inputs['Birth Year'] = pd.to_datetime(user_inputs['Birth Year'], format='%Y').year



# Transform to datetime objects
user_inputs['Accident Date'] = pd.to_datetime(user_inputs['Accident Date'])
user_inputs['Assembly Date'] = pd.to_datetime(user_inputs['Assembly Date'])
user_inputs['C-2 Date'] = pd.to_datetime(user_inputs['C-2 Date'])
user_inputs['C-3 Date'] = pd.to_datetime(user_inputs['C-3 Date'])
user_inputs['First Hearing Date'] = pd.to_datetime(user_inputs['First Hearing Date'])


# hardcode ZIP code 
user_inputs['Zip Code'] = np.nan

# using dictionary to get WCIO and Industry Codes from the descriptions
# C_o_In_dict
# N_o_In_dict
# P_o_B_dict
# Ind_dict


# Transform from description to code for pipeline input
user_inputs['WCIO Cause of Injury Code'] = C_o_In_dict[user_inputs['WCIO Cause of Injury Description']]
user_inputs['WCIO Nature of Injury Code'] = N_o_In_dict[user_inputs['WCIO Nature of Injury Description']]
user_inputs['WCIO Part Of Body Code'] = P_o_B_dict[user_inputs['WCIO Part Of Body Description']]
user_inputs['Industry Code'] = Ind_dict[user_inputs['Industry Code Description']]

# set 'U' value of Alternative Dispute Resolution to np.nan

unknown_values = {'Alternative Dispute Resolution': 'U', 'Alternative Dispute Resolution': 'UNKNOWN',   'Carrier Type': 'UNKNOWN', 'County of Injury': 'UNKNOWN',
    'Gender': 'U','Medical Fee Region': 'UK'}

st.write(user_inputs['Alternative Dispute Resolution'])


for key, value in unknown_values.items():
    if user_inputs[key] == value:

        if key == 'Alternative Dispute Resolution':
            user_inputs[key] = 'N'
        else:
            user_inputs[key] = np.nan

# storing the user inputs in a df
user_inputs_df = pd.DataFrame([user_inputs])


# invert to label again 



# # Display the prediction result
# if prediction[0] == 1:
#     st.write(f"Prediction: Positive Outcome (Risk: {prediction_prob[0]:.2f})")
# else:
#     st.write(f"Prediction: Negative Outcome (Risk: {1 - prediction_prob[0]:.2f})")


# include the array of Outcomes here:


st.write(f'Final prediction: ')



# apply logic to the button
if st.button("Predict"):

    # Load the pipeline from a file
    incoherences_pipeline = joblib.load('../dashboard_objects/incoherence_pipeline.joblib')
    missing_pipeline = joblib.load('../dashboard_objects/missing_values_pipeline.joblib')
    preprocessing_popeline = joblib.load('../dashboard_objects/preprocessing_pipeline.joblib')
    feature_engineering_pipeline = joblib.load('../dashboard_objects/feat_eng_pipeline.joblib')

    # Apply the pipeline to the inputs
    user_inputs_df = incoherences_pipeline.transform(user_inputs_df)
    user_inputs_df = missing_pipeline.transform(user_inputs_df)
    user_inputs_df = preprocessing_popeline.transform(user_inputs_df)
    user_inputs_df = feature_engineering_pipeline.transform(user_inputs_df)



    # read columns to drop from  from txt file
    with open('../dashboard_objects/columns_to_drop.txt', 'r') as f:
        columns_to_drop = f.read().splitlines()



    # drop columns of this input_df
    user_inputs_df = user_inputs_df.drop(columns=columns_to_drop)
    # drop 'C2-Date_Imputed'
    user_inputs_df = user_inputs_df.drop(columns='C-2 Date_Imputed')

    # Here you can integrate your model prediction        
    # Example:
    # load model from pickle file
    # your_model = pickle.load(open("your_model.pkl", ")
    # prediction = your_model.predict(pd.DataFrame([user_inputs]))
    # st.write("Prediction:", prediction)

    # feature_engineering_pipeline = joblib.load('../dashboard_objects/feat_eng_pipeline.joblib')

    # import joblib
    # import joblib model


    final_model = joblib.load('../dashboard_objects/model.joblib')



    # make prediction

    # get the columns that have been used for training 
    model_features = final_model.feature_names_in_

    # make prediction 
    prediction = final_model.predict(user_inputs_df[model_features])



    # use dictionary to invert the prediction
    claim_injury_type_mapping = {
        '4. TEMPORARY': 4-1,
        '2. NON-COMP': 2-1,
        '5. PPD SCH LOSS': 5-1,
        '3. MED ONLY': 3-1,
        '6. PPD NSL': 6-1,
        '1. CANCELLED': 1-1,
        '8. DEATH':8-1,
        '7. PTD': 7-1
    }
    
    # invert the prediction
    inverted_prediction = {v: k for k, v in claim_injury_type_mapping.items()}

    # st.write(inverted_prediction[prediction[0]])

    prediction_string = inverted_prediction[prediction[0]]

    # prediction_prob = final_model.predict_proba(user_inputs_df[model_features])

    # Display the prediction result
    st.write("The Prediction for this indivdual is: ", prediction_string)
    
    # Display the prediction probability
    # st.write(f"Prediction Probability: {prediction_prob[0]:.2f}")

    #st.write(predictions)
    st.write("Disclaimer: This prediction is based on a machine learning model and should not be considered as a final decision")