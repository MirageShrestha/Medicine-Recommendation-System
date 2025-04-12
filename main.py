import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Page configuration
st.set_page_config(
    page_title="Medicine Recommendation System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Medicine Recommendation System\nThis is a machine learning based system for disease prediction and medicine recommendation."
    }
)

# Custom CSS for better styling and larger fonts
st.markdown("""
    <style>
    /* Main content area */
    .main {
        padding: 2rem;
    }
    
    /* Headers */
    h1 {
        font-size: 48px !important;
        font-weight: bold !important;
    }
    h2 {
        font-size: 36px !important;
        font-weight: bold !important;
    }
    h3 {
        font-size: 32px !important;
        font-weight: bold !important;
    }
    
    /* Regular text */
    p, div {
        font-size: 20px !important;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        height: 3.5em;
        font-size: 24px !important;
    }
    
    /* Select boxes and inputs */
    .stSelectbox, .stMultiselect {
        font-size: 22px !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        font-size: 24px !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        font-size: 24px !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-size: 28px !important;
        font-weight: bold !important;
    }
    
    /* Info boxes */
    .stAlert {
        font-size: 20px !important;
    }
    
    /* Lists */
    ul, ol {
        font-size: 20px !important;
    }
    
    /* Custom header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background-color: #f0f8ff;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    /* Disease prediction result */
    .disease-result {
        text-align: center;
        padding: 1.5rem;
        background-color: #e8f4f8;
        border-radius: 10px;
        margin-bottom: 2rem;
        font-size: 32px !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

# Data loading functions
@st.cache_data
def load_data():
    """Load all required datasets"""
    try:
        sym_des = pd.read_csv("symtoms_df.csv")
        precautions = pd.read_csv("precautions_df.csv")
        workout = pd.read_csv("workout_df.csv")
        description = pd.read_csv("description.csv")
        medications = pd.read_csv('medications.csv')
        diets = pd.read_csv("diets.csv")
        return sym_des, precautions, workout, description, medications, diets
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None, None

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        return pickle.load(open('svc.pkl','rb'))
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Helper functions
def get_predicted_value(patient_symptoms, svc):
    """Get disease prediction based on symptoms"""
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

def helper(dis, description, precautions, medications, diets, workout):
    """Get recommendations for a disease"""
    try:
        desc = description[description['Disease'] == dis]['Description']
        desc = " ".join([w for w in desc])

        pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
        pre = [col for col in pre.values]

        med = medications[medications['Disease'] == dis]['Medication'].iloc[0]
        med = med.split(',')

        die = diets[diets['Disease'] == dis]['Diet'].iloc[0]
        die = die.split(',')

        wrkout = workout[workout['disease'] == dis]['workout']
        
        return desc, pre, med, die, wrkout
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return None, None, None, None, None

# Constants
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 
                'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 
                'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 
                'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 
                'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 
                'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 
                'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 
                'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 
                'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 
                'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 
                'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 
                'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 
                'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 
                'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 
                'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 
                'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 
                'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 
                'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 
                'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 
                'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 
                'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 
                'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 
                'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 
                'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 
                'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 
                'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 
                'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 
                'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 
                'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 
                'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 
                'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 
                'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 
                'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 
                'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 
                'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 
                14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 
                17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 
                7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 
                29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 
                19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 
                3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 
                13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 
                26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 
                5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 
                38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Page functions
def home_page():
    """Home page with disease prediction interface"""
    # Main header in a container with custom styling
    with st.container():
        with st.container():
            st.markdown('<div class="main-header">', unsafe_allow_html=True)
            st.title("💊 Medicine Recommendation System")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Load data and model
    sym_des, precautions, workout, description, medications, diets = load_data()
    svc = load_model()
    
    if any(x is None for x in [sym_des, precautions, workout, description, medications, diets, svc]):
        st.error("Failed to load required data or model. Please check the data files and try again.")
        return
    
    # Create a wide container for the main content
    with st.container():
        # Symptoms selection section
        st.header("🔍 Enter Your Symptoms")
        
        # Use a container with custom width for the multiselect
        with st.container():
            selected_symptoms = st.multiselect(
                "Select your symptoms from the list below:",
                list(symptoms_dict.keys()),
                help="You can select multiple symptoms",
                key="symptoms_select"
            )
        
        # Center the diagnosis button
        _, col2, _ = st.columns([2, 1, 2])
        with col2:
            diagnose = st.button("🔍 Get Diagnosis", use_container_width=True, key="diagnose_button")
        
        if diagnose:
            if len(selected_symptoms) > 0:
                with st.spinner("Analyzing symptoms..."):
                    # Get prediction
                    predicted_disease = get_predicted_value(selected_symptoms, svc)
                    
                    # Get recommendations
                    dis_des, precautions_list, medications_list, rec_diet, workout_list = helper(
                        predicted_disease, description, precautions, medications, diets, workout
                    )
                    
                    if any(x is None for x in [dis_des, precautions_list, medications_list, rec_diet, workout_list]):
                        st.error("Failed to get recommendations. Please try again.")
                        return
                    
                    # Display results in a clean layout
                    st.markdown("---")
                    
                    # Disease prediction result
                    with st.container():
                        st.markdown(f"""
                            <div class="disease-result">
                                🎯 Predicted Disease: {predicted_disease}
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Disease description in an expander
                    with st.expander("📝 Disease Description", expanded=True):
                        st.info(dis_des)
                    
                    # Create three columns for recommendations
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Precautions section
                    with st.expander("⚠️ Precautions", expanded=True):
                        for i, precaution in enumerate(precautions_list[0], 1):
                            st.markdown(f"**{i}.** {precaution}")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Medications and Diet in side-by-side expanders
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.expander("💊 Recommended Medications", expanded=True):
                            for i, med in enumerate(medications_list, 1):
                                st.markdown(f"**{i}.** {med}")
                    
                    with col2:
                        with st.expander("🥗 Recommended Diet", expanded=True):
                            for i, diet in enumerate(rec_diet, 1):
                                st.markdown(f"**{i}.** {diet}")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Workout recommendations
                    with st.expander("💪 Recommended Workout", expanded=True):
                        for wrk in workout_list:
                            st.markdown(f"• {wrk}")
            else:
                st.warning("⚠️ Please select at least one symptom")
                
    # Add some spacing at the bottom
    st.markdown("<br><br>", unsafe_allow_html=True)

def about_page():
    """About page with system information"""
    st.header("ℹ️ About the System")
    
    st.write("This is a Medicine Recommendation System that uses machine learning to predict diseases based on symptoms.")
    
    st.subheader("The system provides:")
    
    features = [
        ("🎯 Disease prediction", "Accurate diagnosis based on symptoms"),
        ("💊 Recommended medications", "Appropriate medicines for treatment"),
        ("🥗 Dietary recommendations", "Foods that help in recovery"),
        ("⚠️ Precautions", "Steps to prevent worsening of condition"),
        ("💪 Recommended workout routines", "Exercises to maintain health")
    ]
    
    for title, desc in features:
        st.write(f"**{title}** - {desc}")
    
    st.write("Our system is designed to provide comprehensive health recommendations based on your symptoms.")

def contact_page():
    """Contact page with support information"""
    st.header("📞 Contact Us")
    st.write("For any queries or support, please contact us using the information below:")
    
    with st.container():
        st.subheader("Contact Information")
        contact_info = [
            ("📧 Email", "mirageshrestha7@gmail.com"),
            ("📱 Phone", "9848569098"),
            ("📍 Address", "Balkumari, Lalitpur")
        ]
        
        for icon, info in contact_info:
            st.write(f"**{icon}:** {info}")
    
    st.write("Our support team is available Monday through Friday, 9:00 AM to 5:00 PM EST.")

def developer_page():
    """Developer page with team information"""
    st.header("👨‍💻 Developer Information")
    st.write("This system was developed by a team of medical and technology experts dedicated to improving healthcare accessibility.")
    
    with st.container():
        st.subheader("Technical Support")
        support_info = [
            ("📧 Email", "mirageshrestha7@gmail.com"),
            ("💻 GitHub", "https://github.com/MirageShrestha")
        ]
        
        for icon, info in support_info:
            st.write(f"**{icon}:** {info}")
    
    st.subheader("Our Team")
    team_members = [
        ("👨‍⚕️ Medical Experts", "Providing accurate medical knowledge"),
        ("👨‍💻 Data Scientists", "Developing the prediction algorithms"),
        ("👨‍🎨 UI/UX Designers", "Creating an intuitive interface")
    ]
    
    for role, desc in team_members:
        st.write(f"**{role}** - {desc}")

def blog_page():
    """Blog page with latest updates"""
    st.header("📰 Latest Updates")
    st.write("Check out our latest articles and updates:")
    
    with st.container():
        st.subheader("Featured Articles")
        
        articles = [
            ("📚 Understanding Common Symptoms", "A comprehensive guide to recognizing and understanding common symptoms of various diseases."),
            ("🔬 Latest Medical Research", "Stay updated with the latest breakthroughs in medical research and their implications."),
            ("💡 Health Tips and Advice", "Practical tips and advice for maintaining good health and preventing common ailments."),
            ("🔄 System Updates and Improvements", "Learn about the latest features and improvements to our Medicine Recommendation System.")
        ]
        
        for title, desc in articles:
            st.write(f"**{title}**")
            st.write(desc)

# Main app
def main():
    # Sidebar navigation
    with st.sidebar:
        st.title("🏥 Health Navigator")
        st.markdown("---")
        page = st.radio("Go to", ["🏠 Home", "ℹ️ About", "📞 Contact", "👨‍💻 Developer", "📰 Blog"])
    
    # Page routing
    pages = {
        "🏠 Home": home_page,
        "ℹ️ About": about_page,
        "📞 Contact": contact_page,
        "👨‍💻 Developer": developer_page,
        "📰 Blog": blog_page
    }
    
    pages[page]()

if __name__ == "__main__":
    main()
