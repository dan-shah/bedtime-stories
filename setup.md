# Setup Instructions

## 1. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Set up environment variables
1. Open the `.env` file
2. Replace `your_anthropic_api_key_here` with your actual Anthropic API key
3. Get your API key from: https://console.anthropic.com/

## 4. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Troubleshooting
- Make sure your virtual environment is activated
- Check that your API key is correctly set in the `.env` file
- Ensure all dependencies are installed with `pip list`