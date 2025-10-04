import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import os

# ==================== COPY-PASTE PROTECTION ====================
def disable_copy_paste():
    st.markdown("""
        <style>
        /* Disable text selection */
        * {
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
            user-select: none;
        }
        
        /* Re-enable selection for input fields */
        input, textarea {
            -webkit-user-select: text;
            -moz-user-select: text;
            -ms-user-select: text;
            user-select: text;
        }
        
        /* Hide copy buttons in code blocks */
        .stCodeBlock button[kind="icon"],
        button[title="Copy to clipboard"],
        button[data-testid="stCodeBlockCopyButton"] {
            display: none !important;
        }
        </style>
        
        <script>
        document.addEventListener('contextmenu', function(e) {
            e.preventDefault();
            return false;
        });
        
        document.addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && 
                (e.key === 'c' || e.key === 'C' || 
                 e.key === 'v' || e.key === 'V' || 
                 e.key === 'x' || e.key === 'X' || 
                 e.key === 'a' || e.key === 'A')) {
                e.preventDefault();
                return false;
            }
        });
        
        document.addEventListener('copy', function(e) {
            e.preventDefault();
            return false;
        });
        
        document.addEventListener('cut', function(e) {
            e.preventDefault();
            return false;
        });
        
        document.addEventListener('paste', function(e) {
            e.preventDefault();
            return false;
        });
        
        setTimeout(function() {
            const copyButtons = document.querySelectorAll('button[title="Copy to clipboard"], button[data-testid="stCodeBlockCopyButton"]');
            copyButtons.forEach(button => button.remove());
        }, 500);
        </script>
    """, unsafe_allow_html=True)

# Enable copy protection
disable_copy_paste()

# ==================== STREAMLIT GUIDE ====================
st.title("🚀 Complete Streamlit Guide with Fashion Classifier")

# Sidebar navigation
page = st.sidebar.radio("Choose a section:", 
                        ["Streamlit Basics", "Interactive Widgets Demo", "Data & Charts", 
                         "Code Explanation", "Fashion Classifier"])

# ==================== STREAMLIT BASICS ====================
if page == "Streamlit Basics":
    st.header("📚 Streamlit Basics for Beginners")
    
    st.write("""
    Streamlit is a Python library that turns data scripts into web apps in minutes. 
    No need for HTML, CSS, or JavaScript!
    """)
    
    # Installation
    st.subheader("1️⃣ Installation")
    st.write("Install Streamlit using pip:")
    st.code("pip install streamlit", language="bash")
    
    # Basic Commands
    st.subheader("2️⃣ Text Display Commands")
    st.markdown("""
    **st.title()** - Creates the main title (largest text)  
    **st.header()** - Creates a section header  
    **st.subheader()** - Creates a subsection header  
    **st.write()** - Universal command that displays almost anything  
    **st.markdown()** - Displays text with markdown formatting  
    **st.text()** - Displays fixed-width text  
    """)
    
    st.write("Example:")
    st.code("""st.title("My App")
st.header("Section 1")
st.write("This is some text")""", language="python")
    
    # User Input
    st.subheader("3️⃣ Interactive Widgets")
    
    st.write("**Text Input** - Get text from users")
    name = st.text_input("What's your name?", "Type here...")
    if name:
        st.write(f"Hello, {name}!")
    
    st.write("**Slider** - Let users select a number")
    age = st.slider("Select your age:", 0, 100, 25)
    st.write(f"Selected age: {age}")
    
    st.write("**Select Box** - Dropdown menu")
    option = st.selectbox("Choose an option:", ["Option 1", "Option 2", "Option 3"])
    st.write(f"You selected: {option}")
    
    st.write("**Button** - Clickable button")
    if st.button("Click Me!"):
        st.success("Button was clicked!")
    
    st.write("**File Uploader** - Upload files")
    st.code('uploaded = st.file_uploader("Upload file", type=["jpg", "png"])', 
            language="python")
    
    # Layouts
    st.subheader("4️⃣ Layouts with Columns")
    st.write("Organize content side-by-side using columns:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("Left Column")
        st.write("Content in left column")
    with col2:
        st.warning("Right Column")
        st.write("Content in right column")
    
    st.code("""col1, col2 = st.columns(2)
with col1:
    st.write("Left content")
with col2:
    st.write("Right content")""", language="python")
    
    # Status Messages
    st.subheader("5️⃣ Status Messages")
    st.success("✅ Success - st.success()")
    st.info("ℹ️ Information - st.info()")
    st.warning("⚠️ Warning - st.warning()")
    st.error("❌ Error - st.error()")
    
    # Running App
    st.subheader("6️⃣ Running Your Streamlit App")
    st.write("Save your code in a Python file (e.g., app.py) and run:")
    st.code("streamlit run app.py", language="bash")
    st.write("Your app will automatically open in your web browser!")

# ==================== INTERACTIVE WIDGETS DEMO ====================
elif page == "Interactive Widgets Demo":
    st.header("🎮 Interactive Widgets Demo")
    st.write("Try out all the Streamlit widgets below and see the code!")
    
    st.subheader("📝 Text Input Widgets")
    
    # Text Input
    st.code('''text_input = st.text_input("Label", "Default value")
st.write(f"You typed: {text_input}")''', language="python")
    text_input = st.text_input("Text Input", "Type something here...")
    st.write(f"You typed: {text_input}")
    
    # Text Area
    st.code('''text_area = st.text_area("Label", "Default")
st.write(f"Character count: {len(text_area)}")''', language="python")
    text_area = st.text_area("Text Area (for longer text)", "Write multiple lines here...")
    st.write(f"Character count: {len(text_area)}")
    
    st.markdown("---")
    st.subheader("🔢 Number Input Widgets")
    
    # Number Input
    st.code('number = st.number_input("Label", min_value=0, max_value=100, value=50)', language="python")
    number = st.number_input("Number Input", min_value=0, max_value=100, value=50)
    st.write(f"Selected number: {number}")
    
    # Slider
    st.code('slider = st.slider("Label", 0, 100, 25)', language="python")
    slider_val = st.slider("Slider", 0, 100, 25)
    st.write(f"Slider value: {slider_val}")
    
    # Range Slider
    st.code('range_vals = st.slider("Label", 0, 100, (25, 75))', language="python")
    range_vals = st.slider("Range Slider", 0, 100, (25, 75))
    st.write(f"Selected range: {range_vals[0]} to {range_vals[1]}")
    
    st.markdown("---")
    st.subheader("📋 Selection Widgets")
    
    # Select Box
    st.code('option = st.selectbox("Label", ["Option 1", "Option 2", "Option 3"])', language="python")
    option = st.selectbox("Select Box (dropdown)", 
                         ["Option 1", "Option 2", "Option 3", "Option 4"])
    st.write(f"You selected: {option}")
    
    # Multi Select
    st.code('multi = st.multiselect("Label", ["A", "B", "C"], default=["A"])', language="python")
    multi_options = st.multiselect("Multi Select (choose multiple)",
                                   ["Apple", "Banana", "Cherry", "Date", "Elderberry"],
                                   default=["Apple", "Banana"])
    st.write(f"You selected: {', '.join(multi_options)}")
    
    # Radio Buttons
    st.code('radio = st.radio("Label", ["Yes", "No", "Maybe"])', language="python")
    radio = st.radio("Radio Buttons", ["Yes", "No", "Maybe"])
    st.write(f"You chose: {radio}")
    
    st.markdown("---")
    st.subheader("✅ Boolean Widgets")
    
    # Checkbox
    st.code('agree = st.checkbox("I agree to the terms")', language="python")
    agree = st.checkbox("I agree to the terms")
    if agree:
        st.success("Thank you for agreeing!")
    
    # Toggle
    st.code('toggle = st.toggle("Enable feature")', language="python")
    toggle = st.toggle("Enable feature")
    st.write(f"Feature is: {'ON' if toggle else 'OFF'}")
    
    st.markdown("---")
    st.subheader("🎯 Action Widgets")
    
    # Button
    st.code('if st.button("Click Me!"):\n    st.success("Clicked!")', language="python")
    if st.button("Click Me!"):
        st.balloons()
        st.success("Button clicked! 🎉")
    
    # Download Button
    st.code('''st.download_button(
    label="Download File",
    data="content",
    file_name="file.txt"
)''', language="python")
    sample_text = "This is a sample file content"
    st.download_button(
        label="Download Sample File",
        data=sample_text,
        file_name="sample.txt",
        mime="text/plain"
    )
    
    st.markdown("---")
    st.subheader("📅 Date & Time Widgets")
    
    # Date Input
    st.code('date = st.date_input("Select a date")', language="python")
    date = st.date_input("Select a date")
    st.write(f"Selected date: {date}")
    
    # Time Input
    st.code('time = st.time_input("Select a time")', language="python")
    time = st.time_input("Select a time")
    st.write(f"Selected time: {time}")
    
    st.markdown("---")
    st.subheader("📁 File Upload")
    
    st.code('uploaded = st.file_uploader("Upload file", type=["txt", "jpg"])', language="python")
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "jpg", "png"])
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        st.write(f"File size: {uploaded_file.size} bytes")
    
    st.markdown("---")
    st.subheader("🎨 Color Picker")
    
    st.code('color = st.color_picker("Pick a color", "#00f900")', language="python")
    color = st.color_picker("Pick a color", "#00f900")
    st.write(f"Selected color: {color}")
    st.markdown(f'<div style="width: 100px; height: 100px; background-color: {color}; border-radius: 10px;"></div>', 
                unsafe_allow_html=True)
    
    st.info("💡 Each widget shows its code above - copy the pattern to use in your own apps!")

# ==================== DATA & CHARTS ====================
elif page == "Data & Charts":
    st.header("📊 Working with Data & Charts")
    st.write("Learn how to display data and create visualizations in Streamlit!")
    
    # Sidebar controls
    st.sidebar.subheader("Chart Controls")
    rows = st.sidebar.slider("Number of rows:", 5, 100, 10)
    chart_type = st.sidebar.selectbox("Chart Type", 
                                      ["Line Chart", "Area Chart", "Bar Chart"])
    
    st.subheader("📈 Random Data Example")
    st.write("This example generates random data and displays it in different formats:")
    
    # Code Explanation
    st.code("""import streamlit as st
import pandas as pd
import numpy as np

# Generate random data
rows = st.sidebar.slider("Number of rows:", 5, 100, 10)
data = pd.DataFrame(
    np.random.randn(rows, 3),
    columns=["A", "B", "C"]
)

st.write("Generated Data:", data)
st.line_chart(data)""", language="python")
    
    st.markdown("---")
    st.subheader("🔍 Code Breakdown")
    
    st.markdown("""
    **import pandas as pd** - Library for working with data tables  
    **import numpy as np** - Library for numerical operations  
    
    **np.random.randn(rows, 3)** - Creates random numbers in a rows×3 grid  
    **pd.DataFrame()** - Converts numbers into a structured table  
    **columns=["A", "B", "C"]** - Names the three columns  
    
    **st.write()** - Displays the data table  
    **st.line_chart()** - Creates a line chart from the data  
    """)
    
    st.markdown("---")
    st.subheader("✨ Live Demo")
    
    # Generate data
    data = pd.DataFrame(
        np.random.randn(rows, 3),
        columns=["A", "B", "C"]
    )
    
    # Display data table
    st.write("**Generated Data:**")
    st.dataframe(data, use_container_width=True)
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", rows)
    with col2:
        st.metric("Columns", 3)
    with col3:
        st.metric("Total Values", rows * 3)
    
    # Display chart based on selection
    st.write(f"**{chart_type}:**")
    if chart_type == "Line Chart":
        st.line_chart(data)
    elif chart_type == "Area Chart":
        st.area_chart(data)
    elif chart_type == "Bar Chart":
        st.bar_chart(data)
    
    st.markdown("---")
    st.subheader("📊 More Chart Types")
    
    st.write("**Scatter Plot Example:**")
    scatter_data = pd.DataFrame({
        'x': np.random.randn(50),
        'y': np.random.randn(50),
        'size': np.random.randint(10, 100, 50)
    })
    st.scatter_chart(scatter_data, x='x', y='y', size='size')
    
    st.markdown("---")
    st.subheader("📋 Displaying DataFrames")
    
    st.write("Streamlit offers multiple ways to display data:")
    
    sample_df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'Age': [25, 30, 35, 28],
        'City': ['New York', 'London', 'Tokyo', 'Paris'],
        'Score': [85, 92, 78, 88]
    })
    
    st.write("**st.dataframe()** - Interactive table:")
    st.dataframe(sample_df)
    
    st.write("**st.table()** - Static table:")
    st.table(sample_df)
    
    st.write("**st.metric()** - Display key metrics:")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Average Age", f"{sample_df['Age'].mean():.1f}")
    with metric_col2:
        st.metric("Total People", len(sample_df))
    with metric_col3:
        st.metric("Avg Score", f"{sample_df['Score'].mean():.1f}")
    
    st.info("💡 Use the sidebar slider to change the number of rows in the random data!")

# ==================== CODE EXPLANATION ====================
elif page == "Code Explanation":
    st.header("📖 Fashion Classifier Code Explanation")
    
    st.write("Let's break down each part of the Fashion Classifier code:")
    
    # Section 1: Imports
    st.subheader("Step 1: Import Libraries")
    st.code("""import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os""", language="python")
    
    st.markdown("""
    **streamlit (st)** - Creates the web interface  
    **tensorflow (tf)** - Loads and runs the AI model  
    **PIL (Image)** - Opens and processes images  
    **numpy (np)** - Handles arrays and math operations  
    **os** - Works with file paths and directories  
    """)
    
    # Section 2: Load Model
    st.subheader("Step 2: Load the Pre-trained Model")
    st.code("""working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_fashion_mnist_model.h5"
model = tf.keras.models.load_model(model_path)""", language="python")
    
    st.markdown("""
    **Line 1:** Gets the directory where your Python file is located  
    **Line 2:** Creates the full path to your model file  
    **Line 3:** Loads the trained AI model from the file  
    
    💡 The model must be in the same folder as your Python script!
    """)
    
    # Section 3: Class Names
    st.subheader("Step 3: Define Class Names")
    st.code("""class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']""", language="python")
    
    st.write("""
    This list contains all 10 fashion categories the model can recognize.  
    The model returns a number (0-9), and we use this list to convert it to a readable name.
    """)
    
    # Section 4: Preprocessing Function
    st.subheader("Step 4: Image Preprocessing Function")
    st.code("""def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28, 28))
    img = img.convert('L')
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array""", language="python")
    
    st.markdown("""
    This function prepares the image for the AI model:
    
    **Line 1:** Opens the uploaded image  
    **Line 2:** Resizes to 28x28 pixels (model requirement)  
    **Line 3:** Converts to grayscale (black and white)  
    **Line 4:** Converts to array and normalizes values (0-1 range)  
    **Line 5:** Reshapes to match model input format (1, 28, 28, 1)  
    **Line 6:** Returns the processed image array  
    
    ⚠️ The Fashion MNIST model was trained on 28x28 grayscale images, so we must match that format!
    """)
    
    # Section 5: App Title
    st.subheader("Step 5: Create App Title")
    st.code("st.title('Fashion Item Classifier')", language="python")
    st.write("Displays the main title at the top of your app.")
    
    # Section 6: File Upload
    st.subheader("Step 6: File Upload Widget")
    st.code("""uploaded_image = st.file_uploader("Upload an image...", 
                                   type=["jpg", "jpeg", "png"])""", language="python")
    st.markdown("""
    Creates a file upload button that:
    - Only accepts JPG, JPEG, and PNG files
    - Stores the uploaded file in the `uploaded_image` variable
    - Returns `None` if no file is uploaded
    """)
    
    # Section 7: Main Logic
    st.subheader("Step 7: Process and Classify Image")
    st.code("""if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)
    
    with col1:
        resized_img = image.resize((100, 100))
        st.image(resized_img)
    
    with col2:
        if st.button('Classify'):
            img_array = preprocess_image(uploaded_image)
            result = model.predict(img_array)
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]
            st.success(f'Prediction: {prediction}')""", language="python")
    
    st.markdown("""
    **if uploaded_image is not None:** Only runs if a file is uploaded  
    **image = Image.open():** Opens the uploaded image  
    **col1, col2 = st.columns(2):** Creates 2 columns for layout  
    
    **Left Column (col1):**
    - Resizes image to 100x100 for display
    - Shows the uploaded image
    
    **Right Column (col2):**
    - Creates a "Classify" button
    - When clicked:
      1. Preprocesses the image
      2. Feeds it to the AI model
      3. Gets prediction probabilities
      4. Finds the class with highest probability
      5. Displays the predicted fashion item name
    
    **np.argmax(result):** Finds the index of the highest probability  
    **class_names[predicted_class]:** Converts index to readable name  
    """)
    
    # Flow Diagram
    st.subheader("🔄 How It All Works Together")
    st.markdown("""
    1. **User uploads image** → File uploader widget
    2. **Display image** → Show in left column
    3. **User clicks "Classify"** → Button triggers processing
    4. **Preprocess image** → Resize, grayscale, normalize
    5. **Model prediction** → AI analyzes the image
    6. **Show result** → Display the predicted fashion item
    """)
    
    st.info("💡 The model was trained on the Fashion MNIST dataset with 60,000 grayscale images!")

# ==================== FASHION CLASSIFIER APP ====================
elif page == "Fashion Classifier":
    st.header("👗 Fashion Item Classifier")
    st.write("Upload an image of a fashion item and let AI classify it!")
    
    # Load model
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = f"{working_dir}/trained_fashion_mnist_model.h5"
    
    try:
        model = tf.keras.models.load_model(model_path)
        
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        def preprocess_image(image):
            img = Image.open(image)
            img = img.resize((28, 28))
            img = img.convert('L')
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape((1, 28, 28, 1))
            return img_array
        
        st.info("ℹ️ This model works best with simple, centered images of fashion items on plain backgrounds")
        
        # Show what the model can recognize
        st.write("**The model can recognize these items:**")
        st.write(", ".join(class_names))
        
        uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            col1, col2 = st.columns(2)
            
            with col1:
                resized_img = image.resize((100, 100))
                st.image(resized_img, caption="Uploaded Image")
            
            with col2:
                if st.button('Classify'):
                    with st.spinner('Analyzing image...'):
                        img_array = preprocess_image(uploaded_image)
                        result = model.predict(img_array)
                        predicted_class = np.argmax(result)
                        prediction = class_names[predicted_class]
                        confidence = np.max(result) * 100
                        
                        st.success(f'🎯 Prediction: {prediction}')
                        st.write(f"📊 Confidence: {confidence:.2f}%")
                        
                        # Show all probabilities
                        with st.expander("View all predictions"):
                            for i, class_name in enumerate(class_names):
                                prob = result[0][i] * 100
                                st.write(f"{class_name}: {prob:.2f}%")
    
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Make sure 'trained_fashion_mnist_model.h5' is in the same directory as this script")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("🔒 Copy-paste protection is active")
st.sidebar.write("📚 Navigate between sections using the sidebar")
st.sidebar.write("Built with ❤️ using Streamlit")