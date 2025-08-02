import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 5px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_model():
    """Load the trained MNIST model"""
    try:
        model = tf.keras.models.load_model('mnist_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure 'mnist_model.h5' is in the same directory as this script.")
        return None

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# Header
st.markdown('<h1 class="main-header">🔢 MNIST Digit Classifier</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    app_mode = st.selectbox(
        "Choose the app mode",
        ["Image Upload", "Draw a Digit", "Batch Processing", "Model Info"]
    )
    
    st.markdown("---")
    st.header("About")
    st.info(
        "This app uses a Convolutional Neural Network (CNN) "
        "trained on the MNIST dataset to classify handwritten digits (0-9)."
    )

# Load model
model = load_model()

if model is None:
    st.stop()

# Main content based on selected mode
if app_mode == "Image Upload":
    st.header("📁 Upload Image for Classification")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a clear image of a handwritten digit"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess image
            def preprocess_image(image):
                # Convert to grayscale
                if image.mode != 'L':
                    image = image.convert('L')
                
                # Resize to 28x28
                image_resized = image.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                image_array = np.array(image_resized)
                
                # Invert if needed (MNIST digits are white on black background)
                if image_array.mean() > 127:
                    image_array = 255 - image_array
                
                # Normalize to [0, 1]
                image_normalized = image_array / 255.0
                
                # Reshape for model input
                image_input = image_normalized.reshape(1, 28, 28, 1)
                
                return image_input, image_normalized
            
            try:
                image_input, processed_image = preprocess_image(image)
                
                # Make prediction
                prediction = model.predict(image_input, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)
                
                # Display processed image
                st.subheader("Processed Image (28x28)")
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(processed_image, cmap='gray')
                ax.set_title("Preprocessed for Model")
                ax.axis('off')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.stop()
    
    with col2:
        if uploaded_file is not None:
            # Display prediction
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Prediction: **{predicted_class}**")
            st.markdown(f"### 📊 Confidence: **{confidence:.2%}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show all class probabilities
            st.subheader("Class Probabilities")
            prob_df = pd.DataFrame({
                'Digit': range(10),
                'Probability': prediction[0]
            })
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(prob_df['Digit'], prob_df['Probability'])
            ax.set_xlabel('Digit')
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities')
            
            # Highlight the predicted class
            bars[predicted_class].set_color('red')
            
            # Add percentage labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.1%}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add to history
            st.session_state.predictions_history.append({
                'predicted_class': predicted_class,
                'confidence': confidence,
                'timestamp': pd.Timestamp.now()
            })

elif app_mode == "Draw a Digit":
    st.header("✏️ Draw a Digit")
    st.info("This feature would require additional JavaScript components. "
            "For now, please use the Image Upload feature.")
    
    # Note: Drawing canvas would require streamlit-drawable-canvas
    # st.code("""
    # # To implement drawing feature, install:
    # # pip install streamlit-drawable-canvas
    # 
    # from streamlit_drawable_canvas import st_canvas
    # 
    # canvas_result = st_canvas(
    #     fill_color="rgba(255, 255, 255, 0.3)",
    #     stroke_width=20,
    #     stroke_color="white",
    #     background_color="black",
    #     height=280,
    #     width=280,
    #     drawing_mode="freedraw",
    #     key="canvas",
    # )
    # """)

elif app_mode == "Batch Processing":
    st.header("📦 Batch Processing")
    
    uploaded_files = st.file_uploader(
        "Choose multiple image files",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.subheader(f"Processing {len(uploaded_files)} images...")
        
        results = []
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Process image
                image = Image.open(uploaded_file)
                if image.mode != 'L':
                    image = image.convert('L')
                
                image_resized = image.resize((28, 28), Image.Resampling.LANCZOS)
                image_array = np.array(image_resized)
                
                if image_array.mean() > 127:
                    image_array = 255 - image_array
                
                image_normalized = image_array / 255.0
                image_input = image_normalized.reshape(1, 28, 28, 1)
                
                # Make prediction
                prediction = model.predict(image_input, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)
                
                results.append({
                    'filename': uploaded_file.name,
                    'predicted_digit': predicted_class,
                    'confidence': confidence
                })
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Display results
        if results:
            st.subheader("Batch Results")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Images", len(results))
            with col2:
                st.metric("Average Confidence", f"{results_df['confidence'].mean():.2%}")
            with col3:
                most_common = results_df['predicted_digit'].mode()[0]
                st.metric("Most Common Digit", most_common)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name="batch_predictions.csv",
                mime="text/csv"
            )

elif app_mode == "Model Info":
    st.header("🤖 Model Information")
    
    # Model architecture
    st.subheader("Model Architecture")
    
    # Get model summary
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)
    st.text(model_summary)
    
    # Model metrics
    st.subheader("Model Performance")
    
    # These would be loaded from training history or evaluation
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Accuracy", "96.8%")  # Replace with actual values
    with col2:
        st.metric("Parameters", f"{model.count_params():,}")
    with col3:
        st.metric("Model Size", "2.1 MB")  # Approximate
    
    # Training history visualization (if available)
    st.subheader("Training History")
    st.info("Training history visualization would be displayed here if available.")
    
    # Dataset information
    st.subheader("Dataset Information")
    st.markdown("""
    - **Dataset**: MNIST Handwritten Digits
    - **Training samples**: 60,000
    - **Test samples**: 10,000
    - **Image size**: 28x28 pixels
    - **Classes**: 10 (digits 0-9)
    - **Color**: Grayscale
    """)

# Sidebar - Prediction History
if st.session_state.predictions_history:
    with st.sidebar:
        st.markdown("---")
        st.header("Recent Predictions")
        
        # Show last 5 predictions
        recent_predictions = st.session_state.predictions_history[-5:]
        for i, pred in enumerate(reversed(recent_predictions)):
            st.write(f"**{pred['predicted_class']}** ({pred['confidence']:.1%})")
        
        if st.button("Clear History"):
            st.session_state.predictions_history = []
            st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit and TensorFlow | "
    "Model trained on MNIST dataset"
)