import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys

from sklearn.pipeline import Pipeline

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
    # ("Carrier Name", "object"),
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

# Function to handle user inputs based on column type
def handle_column_input(col, dtype, unique_value_dict):
    user_input = None
    if col in date_columns:
        # Date columns: Provide a checkbox for "Unknown" date and validate input
        unknown_selected = st.checkbox(f"Unknown {col}?", value=False)
        if unknown_selected:
            user_input = np.nan
        else:
            date_input = st.text_input(f"{col} format: (YYYY-MM-DD)", value=pd.Timestamp.now().strftime("%Y-%m-%d"))
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

# Process inputs for Personal Information
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


# storing the user inputs in a df
user_inputs_df = pd.DataFrame([user_inputs])

#applying the pipelines

#incoherences
incoherences_pipeline = Pipeline([
    ('update_carrier_type', IncoCarrierType()),
    ('update_wcio_body_code', IncoWCIOBodyCode()),
    ('replace_birth_year_zero_nan', IncoZeroBirthYEAR()),
    ('replace_age_zero_nan', IncoZeroAgeAtInjury()),
    ('update_dependents', IncoDependents()),
    ('compare_age_with_accident_and_birth', IncoCorrectAge()),
    ('swap_accident_date', IncoSwapAccidentDate()), 
    ('update_covid_indicator', IncoCovidIndicator()),
    ('replace_gender_x_to_nan',IncoGenderNaN())
])

# Get WCIO columns
columns_code = ['WCIO Part Of Body Code', 'WCIO Cause of Injury Code', 'WCIO Nature of Injury Code']
columns_desc = ['WCIO Part Of Body Description', 'WCIO Cause of Injury Description', 'WCIO Nature of Injury Description']

# Create the pipeline with the preprocessor and custom transformers
missing_pipeline = Pipeline([
    ('fill_ime4', FillNaNValues(column='IME-4 Count', fill_value=0)),  # Custom transformer for 'IME-4 Count'
    ('fill_zip_code', FillNaNValues(column='Zip Code', fill_value='UNKNOWN')),
    ('impute_birth_year_from_accident', ImputeBirthYearFromAccident()),
    ('impute_birth_year_with_median_age_and_birth', ImputeBirthYearWithMedian()),
    ('impute_medical_fee_region', ImputeProportionalTransformer(column='Medical Fee Region')),
    ('impute_industry_code', ImputeProportionalTransformer(column='Industry Code')),
    ('fill_missing_descriptions', FillMissingDescriptionsWithCode(
        code_column='Industry Code', description_column='Industry Code Description')),
    # ('impute_accident_date_with_assembly', ImputeAccidentDate()),
    ('impute_age_at_injury',ImputeAgeAtInjury()),
    ('impute_alternative_dispute_resolution', ImputeProportionalTransformer(column='Alternative Dispute Resolution')),
    ('impute_all_wcio_missing_with_unknown', ImputeWithUnknownWCIO(columns_code=columns_code, columns_desc=columns_desc)),
    ('impute_wcio_part_of_body_code', ImputeUsingModeAfterGrouping(
        grouping_column='WCIO Cause of Injury Code', column_to_impute='WCIO Part Of Body Code')),
    ('fill_missing_descriptions_part_of_body', FillMissingDescriptionsWithMapping(
        code_column='WCIO Part Of Body Code', description_column='WCIO Part Of Body Description')),
    ('impute_wcio_cause_of_injury_code', ImputeUsingModeAfterGrouping(
        grouping_column='WCIO Part Of Body Code', column_to_impute='WCIO Cause of Injury Code')),
    ('fill_missing_descriptions_cause_of_injury', FillMissingDescriptionsWithMapping(
        code_column='WCIO Cause of Injury Code', description_column='WCIO Cause of Injury Description')),
    ('impute_wcio_nature_of_injury_code', ImputeUsingModeAfterGrouping(
        grouping_column='WCIO Part Of Body Code', column_to_impute='WCIO Nature of Injury Code')),
    ('fill_missing_descriptions_nature_of_injury', FillMissingDescriptionsWithMapping(
        code_column='WCIO Nature of Injury Code', description_column='WCIO Nature of Injury Description')),
    ('impute_gender', ImputeProportionalTransformer(column='Gender')),
    ('impute_carrier_type', ImputeProportionalTransformer(column='Carrier Type')),
    ('impute_county_of_injury', ImputeProportionalTransformer(column='County of Injury')),
    ('fill_aww_with_nys_aww',  FillNaNValues(column='Average Weekly Wage', fill_value=1757.19)),
    ('impute_c2_date_with_avg_between_accident', ImputeC2Date()),
])

#preprocessing
# define binary columns and target encoder columns
binary_columns_list = ['Alternative Dispute Resolution'
                       , 'COVID-19 Indicator',
                       'Attorney/Representative',]
target_encoder_list = ['Industry Code',
                    'WCIO Cause of Injury Code',
                    'WCIO Nature of Injury Code',
                    'WCIO Part Of Body Code']


# Define the mapping for the carrier type
carrier_type_mapping = {
    '1A. PRIVATE': 'Private Insurance Carrier',
    '2A. SIF': 'State Insurance Fund',
    '3A. SELF PUBLIC': 'Self-insured Public Entity',
    '4A. SELF PRIVATE': 'Self-insured Private Entity',
    '5A. SPECIAL FUND - CONS. COMM. (SECT. 25-A)': 'Special Funds',
    '5C. SPECIAL FUND - POI CARRIER WCB MENANDS': 'Special Funds',
    '5D. SPECIAL FUND - UNKNOWN': 'Special Funds',
    'UNKNOWN': 'Unknown'
}
preprocessing_popeline = Pipeline(steps=[
     # Apply the binary  transformation for the list 
    ('binary_encoder_bin', BinaryEncoder(binary_columns=binary_columns_list)),

    # Apply the target encoder for following columns
    ('target_encoder_in' ,  MultipleTargetEncoder(feature_column = 'Industry Code')),
    ('target_encoder_coi', MultipleTargetEncoder(feature_column = 'WCIO Cause of Injury Code')),
    ('target_encoder_noi', MultipleTargetEncoder(feature_column = 'WCIO Nature of Injury Code')),
    ('target_encoder_pob', MultipleTargetEncoder(feature_column = 'WCIO Part Of Body Code')),
            
    # map column with mapping_dict to new structure, drop = true means that the new created column will replace the original
    ('mapper_carrier_type', ColumnMapper(column_name = 'Carrier Type', mapping_dict = carrier_type_mapping, drop_original = True)),

    # Use one Hot encoder/create dummies for the following 
    # ('dummy_encoder', DummyEncoder(dummy_column = 'Attorney/Representative')), # rewrote it with one hot encoding !!!
    ('dummy_encoder_Carrier_Type', DummyEncoder(dummy_column = 'Carrier Type')), # rewrote it with one hot encoding !!!
    
    # encode na as 1 rest as 0
    ('na_indicator_C3', NAIndicatorEncoder('C-3 Date')), 
    ('wage misisng', NAIndicatorEncoder('Average Weekly Wage')),
    ('na_indicator_C4', NAIndicatorEncoder('First Hearing Date')), 
    ])

#feature engineering
feature_engineering_pipeline = Pipeline(

    [
    # Encode Accident Date as season with Spring, Autumn etc.
    ('season_transformer', SeasonTransformer(date_column = 'Accident Date')),
    # Apply DummyEncoder to the new season column
    ('dummy_encoder_season', DummyEncoder(dummy_column = 'Accident Date_Season')),
     

    # create a new column that calculates days between two columns and then apply log transformation 
    ('days_between_acc_ass', Days_between(start_col = 'Accident Date', end_col = 'Assembly Date')), # output date is called 'Days_between_{end_col}_{start_col}
    ('days_between_acc_ass_log', LogTransformer(column = 'Days_between_Assembly Date_Accident Date')), # transforms column with log here ln()


    # create a new column that calculates days between two columns and then apply log transformation 
    ('days_between_acc_C2', Days_between(start_col = 'Accident Date', end_col = 'C-2 Date' )), # output date is called 'Days_between_{end_col}+{start_col}'
    ('days_between_acc_C2_log', LogTransformer(column = 'Days_between_C-2 Date_Accident Date',  )), # transforms column with log here ln()

    # apply log transformation for average weekly wage 
    ('average_weekly_wage_log', LogTransformer(column = 'Average Weekly Wage')), # transforms column with log here ln()

    # Encode income with quantiles
    # ('IncomeCategorization', CategorizeIncomeDescriptive()), # I think it doesnt make that much sense to apply. That way we get three new columns with onehotencoding
    
    # NumberBining for Age at Injury
    # ('AgeBinning', NumberBinning(init_col_name = 'Age at Injury',  column_name = 'Age Group')), # Keep it as numerical for now we can try this later on
    ]
)

# st.write(user_inputs_df.columns) # seeing the columns
st.write(user_inputs)

''' CHECKING WHAT COLUMNS WE ARE GETTING
# Before running predictions, check if columns exist and are transformed correctly
required_columns = ['Accident Date', 'Assembly Date', 'Average Weekly Wage', 'Gender', 'Number of Dependents', 
                    'Age at Injury', 'Carrier Type', 'WCIO Part Of Body Code', 'WCIO Cause of Injury Code',
                    'WCIO Nature of Injury Code', 'WCIO Part Of Body Description', 'WCIO Cause of Injury Description',
                    'WCIO Nature of Injury Description', 'Medical Fee Region', 'County of Injury', 'Attorney/Representative']

# Check if any of the required columns are missing
missing_columns = [col for col in required_columns if col not in user_inputs_df.columns]
if missing_columns:
    st.write(f"Warning: Missing columns: {missing_columns}")
else:
    st.write("All required columns are present.")'''

# Apply the pipeline to the inputs
user_inputs_df = incoherences_pipeline.transform(user_inputs_df)
user_inputs_df = missing_pipeline.transform(user_inputs_df)
user_inputs_df = preprocessing_popeline.transform(user_inputs_df)
user_inputs_df = feature_engineering_pipeline.transform(user_inputs_df)



# Make predictions using the pre-trained model
prediction = model.predict(user_input)
prediction_prob = model.predict_proba(user_input)[:, 1]
# Display the prediction result
if prediction[0] == 1:
    st.write(f"Prediction: Positive Outcome (Risk: {prediction_prob[0]:.2f})")
else:
    st.write(f"Prediction: Negative Outcome (Risk: {1 - prediction_prob[0]:.2f})")


# apply logic to the button
if st.button("Predict"):
    # Here you can integrate your model prediction
    # Example:
    # load model from pickle file
    # your_model = pickle.load(open("your_model.pkl", ")
    # prediction = your_model.predict(pd.DataFrame([user_inputs]))
    # st.write("Prediction:", prediction)

    st.write("Prediction placeholder:")
   #st.write("Please implement your prediction logic here.")
    #st.write(predictions)
    st.write("Disclaimer: ")