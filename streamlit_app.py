"""
Mental Health Detector - Streamlit Cloud Entry Point
This file is specifically designed for Streamlit Cloud deployment.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="Mental Health Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add current directory to Python path for imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Try to import and run the main app
try:
    # Import the main app module
    import app
    
    # The app.py file contains the main() function that runs the Streamlit app
    # Since we're importing it, it will execute automatically
    
except ImportError as e:
    st.error("🚨 Import Error")
    st.error(f"Failed to import required modules: {e}")
    
    st.markdown("### 🔧 Troubleshooting Steps:")
    st.markdown("""
    1. **Check Dependencies**: Ensure all packages in `requirements.txt` are installed
    2. **Verify File Structure**: Make sure all source files are present
    3. **Model Setup**: Run `python setup_models.py` to download AI models
    4. **Python Path**: Verify the project structure is correct
    """)
    
    st.markdown("### 📁 Expected Project Structure:")
    st.code("""
    Mental-Health-Detector-AI-Application-Built/
    ├── streamlit_app.py (this file)
    ├── app.py (main application)
    ├── requirements.txt
    ├── setup_models.py
    └── src/
        ├── models/
        ├── utils/
        └── cli/
    """)
    
    st.markdown("### 🆘 Need Help?")
    st.markdown("""
    - 📖 [Documentation](https://github.com/midlaj-muhammed/Mental-Health-Detector-AI-Application-Built#readme)
    - 🐛 [Report Issues](https://github.com/midlaj-muhammed/Mental-Health-Detector-AI-Application-Built/issues)
    - 💬 [Discussions](https://github.com/midlaj-muhammed/Mental-Health-Detector-AI-Application-Built/discussions)
    """)
    
    st.stop()

except Exception as e:
    st.error("🚨 Application Error")
    st.error(f"An unexpected error occurred: {e}")
    
    with st.expander("🔍 Error Details"):
        st.code(str(e))
        
    st.markdown("### 🆘 Crisis Resources")
    st.markdown("""
    **If you're experiencing a mental health crisis:**
    - 🚨 **Emergency**: 911
    - 📞 **National Suicide Prevention Lifeline**: 988
    - 💬 **Crisis Text Line**: Text HOME to 741741
    """)
    
    st.stop()
