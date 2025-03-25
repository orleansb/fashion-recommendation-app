import numpy as np
import pandas as pd
import streamlit as st
import joblib
import json
import os
import base64
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import gspread
from google.oauth2 import service_account

#Path to your background image
background_image = "static/back_images.jpg"

# Convert the image to a base64-encoded string
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded_string}"

# Get the base64-encoded image
background_image_base64 = get_base64_image(background_image)

# Custom CSS to set the background image
background_style = f"""
<style>
.stApp {{
    background-image: url("{background_image_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
</style>
"""

# Inject the custom CSS into the app
st.markdown(background_style, unsafe_allow_html=True)


# Load trained model
model = joblib.load("model_file/kmeans_model_hh.pkl")

# Load scalers and encoder
min_max_scaler = joblib.load("model_file/minmax_scaler.pkl")
scaler = joblib.load("model_file/scaler (3).pkl")
encoder = joblib.load("model_file/label_encoder.pkl")
product_reduced = np.load("model_file/similarity_matrix.npy")
similarity_matrix = np.load("model_file/similarity_matrix.npy")
prod_scaler = joblib.load("model_file/prod_minmax_scaler.pkl")
product_data = pd.read_csv("model_file/BrandProductsGendered.csv")

st.title("Fashion Recommendation System")

# Create input fields for user testing
age = st.number_input("Enter Age", min_value=10, max_value=100)

gender = st.selectbox("Select Gender", ["Male", "Female", "Prefer not to say"])
tribe = st.selectbox("Select Tribe", ["Akan", "Ewe", "Ga", "Fante", "Yoruba", "Krobo", "Other"])
occupation = st.selectbox("Select Occupation", ["Student", "Employed", "Doctor", "Engineer", "Artist", "Teacher", "Other"])

favorite_color = st.selectbox("Select Favorite Color", [
    "Red", "Pink", "Turquoise", "Yellow", "Orange", "Brown", 
    "Purple", "Green", "Blue", "Black", "White", "Navy Blue", 
    "Other"
])

fashion_style = st.selectbox("Select Fashion Style", ["Minimalist", "Sportswear", "Bohemian", "Streetwear", "Casual", "Formal", "Other"])

garment_fitting = st.selectbox("Select Garment Fitting", ["Tight-fitting", "Slim Fit", "Oversized", "Baggy", "Loose-fitting", "Regular-fitting", "Structured", "Flowy", "Classic Fit", "Regular Fit", "Other"])

trouser_style = st.selectbox("Select Trouser Style", [
    "Cargo Trouser", "Bootcut", "Jogger", "Straight-cut", "Skinny", 
    "Flared Leg Trouser", "Low Waist", "High Rise", "Wide-leg", "Other"
])

neck_style = st.selectbox("Select Neck Style", ["Crewneck", "Square Neck", "Boatneck", "Collars", "High Neck", "Round Neck", "V Neck", "Sweetheart", "Other"])

cloth_options = ["Shirt", "Skirt", "Trousers", "Dresses", "T-shirt"]
cloth_type = st.multiselect("Select Cloth Type", cloth_options)

def get_cloth_dummies(selected_cloths, cloth_options):
    return np.array([[1 if cloth in selected_cloths else 0 for cloth in cloth_options]])

encoded_cloth = get_cloth_dummies(cloth_type, cloth_options)

hobbies = st.selectbox("Select Hobbies", [
    "Educational and Intellectual Hobbies", "Social and Recreational Hobbies", "Collecting and DIY Hobbies", 
    "Cooking", "Creative and Artistic Hobbies", "Physical and Outdoor Hobbies", "Other"
])

color = st.selectbox("Select Color Preference", [
    "Dark Tones", "Bright Colors", "Neutral Tones", "Earthy Tones", "Pastels", "Patterns and Prints", "Other"
])

# Personality Traits (Big 5)
st.subheader("Openness to Experience")
openness = np.mean([
    st.slider("I am willing to try new fashion trends.", 0, 5, 3),
    st.slider("I enjoy experimenting with my clothing choices and styles.", 0, 5, 3),
    st.slider("I am creative and innovative in my clothing choices.", 0, 5, 3),
    st.slider("I like to explore unique and artistic fashion items.", 0, 5, 3),
    st.slider("I appreciate fashion that challenges traditional norms.", 0, 5, 3)
])

st.subheader("Conscientiousness")
conscientiousness = np.mean([
    st.slider("I prefer a well-organized wardrobe with a limited selection of outfits.", 0, 5, 3),
    st.slider("Punctuality and being on time for appointments are important to me.", 0, 5, 3),
    st.slider("I am focused on achieving my fashion goals and presenting a professional image.", 0, 5, 3),
    st.slider("I follow fashion trends and keep up with the latest fashion news.", 0, 5, 3),
    st.slider("I am disciplined and detail-oriented in my clothing choices.", 0, 5, 3)
])

st.subheader("Extraversion")
extraversion = np.mean([
    st.slider("I enjoy dressing up for social events and gatherings.", 0, 5, 3),
    st.slider("I like being the center of attention with my clothing choices.", 0, 5, 3),
    st.slider("I enjoy shopping with friends.", 0, 5, 3),
    st.slider("I am outgoing and expressive in my fashion choices.", 0, 5, 3),
    st.slider("I like to make a fashion statement in public.", 0, 5, 3)
])

st.subheader("Agreeableness")
agreeableness = np.mean([
    st.slider("I prioritize comfort and practicality when choosing my clothing.", 0, 5, 3),
    st.slider("Creating a positive impression through clothing is important to me.", 0, 5, 3),
    st.slider("I prefer clothing that makes me approachable and friendly.", 0, 5, 3),
    st.slider("I tend to follow fashion choices that others around me prefer.", 0, 5, 3),
    st.slider("Harmony and cooperation are important considerations in my fashion choices.", 0, 5, 3)
])

st.subheader("Neuroticism")
neuroticism = np.mean([
    st.slider("Emotional comfort and confidence greatly influence my clothing choices.", 0, 5, 3),
    st.slider("I am concerned about how others perceive me based on my clothing.", 0, 5, 3),
    st.slider("I change my clothing choices based on my emotional state.", 0, 5, 3),
    st.slider("Clothing issues can affect my emotions.", 0, 5, 3),
    st.slider("I use clothing as a means of self-expression and emotional release.", 0, 5, 3)
])

predicted_cluster =[]

# Ensure session state variables are initialized
if "predicted_cluster" not in st.session_state:
    st.session_state.predicted_cluster = None

# Button to submit inputs
if st.button("Predict Clothing Preferences"):

    categorical = np.array([[gender, occupation, hobbies, favorite_color, fashion_style, garment_fitting, trouser_style, neck_style, color, tribe]])
    numerical = np.array([[age, openness, conscientiousness, extraversion, agreeableness, neuroticism]])

    categorical_col = [] 
    for col in categorical:
        cat_col = encoder.fit_transform(col) 
        categorical_col.append(cat_col)

    num_col = scaler.fit_transform(numerical)  

    categorical_col = np.array(categorical_col).reshape(1, -1)  
    num_col = np.array(num_col).reshape(1, -1)  
    encoded_cloth = np.array(encoded_cloth).reshape(1, -1)  

    user_input = np.hstack([categorical_col, num_col, encoded_cloth])

    # Scale the input
    user_input_scaled = min_max_scaler.transform(user_input)

    # Make prediction
    prediction = model.predict(user_input_scaled)

    cluster_names = {
    0: """Elegant Minimalists  
    You prefer clean, classic, and timeless fashion.  
    Your wardrobe consists of neutral colors, fitted garments, and smart-casual or formal wear.  
    Functionality and simplicity define your style.""",
        
        1: """Trendy Fashionistas  
    You love bold colors, trendy outfits, and statement pieces.  
    Fashion is your form of self-expression, and you stay updated with the latest trends through social media and influencers.""",
        
        2: """Comfort-First Casuals  
    You prioritize comfort over trends.  
    Your go-to outfits include T-shirts, relaxed-fit trousers, and simple, durable clothing.  """,
        
        3: """Bold Statement Makers  
    You enjoy making an impact with your fashion choices.  
    Structured garments, bold colors, and unique styles help you stand out.  
    Fashion is a key part of your identity.""",
        
        4: """Bohemian Free Spirits  
    You embrace flowy fabrics, earthy tones, and unconventional styles.  
    Fashion for you is about creativity, comfort, and self-expression with a relaxed, artistic vibe.""",
        
        5: """Sporty Athleisure Enthusiasts  
    You prefer sporty, functional outfits like track pants, sneakers, and performance fabrics.  
    Whether at the gym or out and about, your style is active, comfortable, and effortless."""
    }

    st.session_state.predicted_cluster = prediction[0]  
    predicted_name = cluster_names.get(st.session_state.predicted_cluster, "Unknown Cluster")

    # Display results
    st.success(predicted_name) 

# Button to get recommendations
if st.button("Get Recommendations"):

    if st.session_state.predicted_cluster is None:
        st.warning("Please predict clothing preferences first.")
    else:
        def recommend_products_for_cluster(cluster_id, top_n=4):

            if gender == "Male":
                filtered_products = product_data[product_data["Gender"].isin(["Unisex"])]
            elif gender == 'Female':
                filtered_products = product_data[product_data['Gender'].isin(["Female", "Unisex"])]
            else:
                filtered_products = product_data

            if filtered_products.empty:
                return pd.DataFrame(columns=["Product Name", "Image"])  # Return an empty DataFrame


            product_indices = similarity_matrix[cluster_id].argsort()[-top_n:][::-1]

            product_indices = [i for i in product_indices if i < len(filtered_products)]
            if not product_indices:  # If no valid indices exist
                return pd.DataFrame(columns=["Product Name", "Image"])

            return filtered_products.iloc[product_indices][['Product Name', 'Image']]

        cluster_id = st.session_state.predicted_cluster

        recommendations = recommend_products_for_cluster(cluster_id)

        st.write("### Recommended Products for You:")

        for _, row in recommendations.iterrows():
                product_name = row["Product Name"]
                image_filename = row["Image"]
                image_filename = os.path.basename(image_filename)  # Extract just the filename
                

                # Construct the full image path
                image_path = os.path.join("static", image_filename)
                
                st.write(f"**{product_name}**")  # Display the product name
                
                # Check if the image exists and display it
                if os.path.exists(image_path):
                    st.image(image_path, caption=product_name, use_column_width=False, width=200)  # Adjust width as needed
                else:
                    st.write("Image not found")  # Debugging: Indicate if the image is missing

# Add a section for rating the prediction accuracy
# Function to save responses to Google Sheets
def save_to_google_sheets(data):
   try:
        # 1. Verify secrets exist
        if 'google_credentials' not in st.secrets:
            st.error("❌ Google credentials not found in secrets!")
            return False

        creds_data = st.secrets['google_credentials']
        
        # 2. Convert credentials to proper format
        if isinstance(creds_data, str):
            try:
                # Try parsing as JSON string
                creds = json.loads(creds_data)
            except json.JSONDecodeError:
                # If string but not JSON, try evaluating (careful with security)
                try:
                    creds = eval(creds_data)  # Only safe if you control the secrets
                except:
                    st.error("❌ Credentials string is not valid JSON or Python dict")
                    return False
        elif isinstance(creds_data, dict):
            creds = creds_data
        else:
            st.error("❌ Credentials must be either JSON string or dictionary")
            return False

        # 3. Validate credential structure
        required_keys = {
            'type', 'project_id', 'private_key_id', 
            'private_key', 'client_email', 'client_id'
        }
        if not all(key in creds for key in required_keys):
            st.error("❌ Missing required credential fields")
            return False

        # 4. Authorize and save to sheets
        try:
            credentials = service_account.Credentials.from_service_account_info(creds)
            scope = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
            client = gspread.authorize(credentials.with_scopes(scope))
            sheet = client.open("Streamlit-Responses").sheet1
            sheet.append_row(data)
            st.success("✅ Data saved successfully!")
            return True
        except MalformedError:
            st.error("❌ Invalid credential format (malformed private key)")
        except gspread.exceptions.APIError as e:
            st.error(f"❌ Sheets API error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
        
        return False

    except Exception as e:
        st.error(f"❌ Critical error: {str(e)}")
        return False

# Add a section for rating the prediction accuracy
if st.session_state.predicted_cluster is not None:
    st.subheader("Rate the Accuracy of the Prediction")
    accuracy_rating = st.slider("How accurate was the prediction? (1 = Not Accurate, 5 = Very Accurate)", 1, 5, 3)

    if accuracy_rating < 4:  # If the user feels the prediction was not accurate
        st.write("If the prediction was not accurate, please select the personality group that you believe fits you better:")
        
        # Display the cluster names for the user to choose from
        cluster_names = {
            0: "Elegant Minimalists",
            1: "Trendy Fashionistas",
            2: "Comfort-First Casuals",
            3: "Bold Statement Makers",
            4: "Bohemian Free Spirits",
            5: "Sporty Athleisure Enthusiasts"
        }
        
        selected_cluster = st.selectbox("Select a Personality Group", list(cluster_names.items()), format_func=lambda x: x[1])
        
        # Get the cluster ID from the selected option
        selected_cluster_id = selected_cluster[0]
        
        # Display products for the selected cluster
        st.write(f"### Recommended Products for {cluster_names[selected_cluster_id]}:")
        
        def recommend_products_for_cluster(cluster_id, top_n=4):
            if gender == "Male":
                filtered_products = product_data[product_data["Gender"].isin(["Unisex"])]
            elif gender == 'Female':
                filtered_products = product_data[product_data['Gender'].isin(["Female"])]
            else:
                filtered_products = product_data

            if filtered_products.empty:
                return pd.DataFrame(columns=["Product Name", "Image"])  # Return an empty DataFrame

            product_indices = similarity_matrix[cluster_id].argsort()[-top_n:][::-1]

            product_indices = [i for i in product_indices if i < len(filtered_products)]
            if not product_indices:  # If no valid indices exist
                return pd.DataFrame(columns=["Product Name", "Image"])

            return filtered_products.iloc[product_indices][['Product Name', 'Image']]

        recommendations = recommend_products_for_cluster(selected_cluster_id)

        for _, row in recommendations.iterrows():
            product_name = row["Product Name"]
            image_filename = row["Image"]
            image_filename = os.path.basename(image_filename) 

            image_path = os.path.join("static", image_filename)
            
            st.write(f"**{product_name}**")  # Display the product name
            
            if os.path.exists(image_path):
                st.image(image_path, caption=product_name, use_column_width=False, width=200)  # Adjust width as needed
            else:
                st.write("Image not found")  # Debugging: Indicate if the image is missing

        preference = st.radio("Do you prefer these clothes?", ("Yes", "No"))
        
        if preference == "Yes":
            st.success("Great! We're glad you found something you like.")
        else:
            st.warning("We're sorry to hear that. Please try again or provide more details for better recommendations.")

        # Save the response to Google Sheets
        response_data = [
            age, gender, tribe, occupation, favorite_color, fashion_style, garment_fitting,
            trouser_style, neck_style, ", ".join(cloth_type), hobbies, color, openness,
            conscientiousness, extraversion, agreeableness, neuroticism, accuracy_rating,
            selected_cluster_id, preference
        ]
        save_to_google_sheets(response_data)
