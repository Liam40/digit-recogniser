import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import time
from datetime import datetime
from torch import load
import database_utils
import pandas as pd
import torch
from train import Net
from torchvision import transforms
import torch.nn.functional as F


torch.classes.__path__ = []

# Start time for runtime statistics
start_time = time.time()

# Function to get ordinal suffix for the day
def get_ordinal_suffix(day):
    if 4 <= day <= 20 or 24 <= day <= 30:
        return "th"
    return ["st", "nd", "rd"][(day % 10) - 1] if day % 10 in [1, 2, 3] else "th"

# Get current date and time
now = datetime.now()
current_time = now.strftime(f"%d{get_ordinal_suffix(now.day)} %B %Y, %I:%M %p")

# Streamlit App Title
st.title("Digit Recognizer")
st.subheader(current_time)

# Initialize database (error handling is done in database_utils)
database_utils.initialise_database()
model = Net()
model.load_state_dict(torch.load('model.pth', weights_only=True))
model.eval()

predicted_label = 0
confidence = 0

# Create layout with two columns
col1, col2 = st.columns(2)

# Column 1: Drawing Canvas
with col1:
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",
        stroke_width=15,
        stroke_color="#D3D3D3",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True,
    )
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        img = img.convert("L") 
        img_transform = transforms.Compose([
                transforms.Resize((28, 28)),  # Adjust size as needed
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # Adjust normalization as needed
            ])
        img_tensor = img_transform(img).unsqueeze(0)
        output = model(img_tensor)

        probabilities = F.softmax(output, dim=1)  # Apply softmax to get probabilities
        predicted_label = torch.argmax(probabilities, dim=1).item()  # Get argmax to the prediction
        confidence = probabilities[0][predicted_label].item()  # Get the max probabilities




# Column 2: Labels and Inputs
with col2:
    print(predicted_label)
    st.text_input("Prediction", value=predicted_label, key="prediction", disabled=True)
    st.text_input("Confidence", value=confidence, key="confidence", disabled=True)
    # True Label Input (user-provided)
    true_label = st.text_input("True Label", value="0", max_chars=1, key="true_label")


    # Submit Button Styling
    st.markdown(
        """
        <style>
        div.stButton > button {
            background-color: #90EE90;
            color: #000000;
            font-size: 20px;
            height: 3em;
            width: 100%;
            border-radius: 10px;
            border: 2px solid #000000;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Submit Button Logic
    if st.button("Submit") and true_label != "0":
        database_utils.save_to_database(now, predicted_label, true_label)

# Fetch and Display History Table
st.markdown("---")
st.markdown("### History Table")

history = database_utils.fetch_history()
if history:
    df = pd.DataFrame(history, columns=["Timestamp", "Prediction", "Label"])
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.write("No records found.")

# Calculate and Display Runtime Statistics
st.markdown("---")
st.markdown("### Runtime Statistics")
st.write(f"Total execution time: {time.time() - start_time:.4f} seconds")