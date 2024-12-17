import streamlit as st
import pandas as pd
import numpy as np

# Assume data is already loaded into a dataframe called `data`
# data = pd.read_csv("../project_data/train_data.csv", dtype={'Zip Code': str})  # for example

st.title("Prediction Interface")
st.write("Please select values for each feature to get an instant prediction.")

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

user_inputs = {}

for col, dtype in columns_info:
    if col in date_columns:
        # Date columns: Use text input for manual entry in YYYY-MM-DD format
        date_input = st.text_input(f"{col} format: (YYYY-MM-DD)", value=pd.Timestamp.now().strftime("%Y-%m-%d"))
        if date_input.strip().lower() == "unknown" or date_input.strip() == "":
            user_inputs[col] = np.nan
    elif dtype == "object":
        # Object column
        unique_values = unique_value_dict[col]
        unique_values = list(unique_values)
        # Add "Unknown" option at the beginning
        unique_values = ["Unknown"] + unique_values
        selected = st.selectbox(f"{col}", unique_values)
        if selected == "Unknown":
            user_inputs[col] = np.nan
        else:
            user_inputs[col] = selected
    else:
        # Numeric column
        # Identify which ones should be int
        if col == "Age at Injury":
            # Integer input
            # valid_values = data[col].dropna()
            default_val = 30
            user_inputs[col] = int(st.number_input(
                label=f"{col}",
                value=default_val,
                step=1
            ))
        elif col == "IME=4 Count":
            default_val = 0
            user_inputs[col] = int(st.number_input(
                label=f"{col}",
                value=default_val,
                step=1
            ))
        else:
            # Other numeric columns remain float
            # valid_values = data[col].dropna()
            default_val = 1
            user_inputs[col] = int(st.number_input(
                label=f"{col}",
                value=default_val,
                step=1
            ))

if st.button("Predict"):
    # Here you can integrate your model prediction
    # Example:

    # load model from pickle file
    # your_model = pickle.load(open("your_model.pkl", ")
    # prediction = your_model.predict(pd.DataFrame([user_inputs]))
    # st.write("Prediction:", prediction)

    st.write("Prediction placeholder:")
    st.write("Please implement your prediction logic here.")