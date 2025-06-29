def select_theme(is_dark_mode):
    if is_dark_mode:
        return """
            html, body, .stApp {
                background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
                color: #f1f1f1;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .stTextInput input, .stFileUploader, .stButton > button {
                background-color: #1e1e1e;
                color: #f1f1f1;
            }
            .stTextInput input::placeholder {
                color: #cc8899 !important;
                opacity: 0.8;
            }
            .stFileUploader {
                border: 2px dashed #4a90e2;
                padding: 1rem;
                background-color: #1e1e1e;
                border-radius: 10px;
            }
            .stFileUploader > label {
                color: #ffffff !important;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: #2a2a2a;
                border-radius: 8px 8px 0 0;
                font-size: 16px;
                font-weight: bold;
                color: #f1f1f1;
                padding: 1rem;
                margin-right: 3px;
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                background-color: #4a90e2;
                color: white;
                font-weight: bold;
                box-shadow: 0 4px 6px rgba(0,0,0,0.5);
            }
            .sample-question-button button {
                background-color: #2a2a2a;
                border: 2px solid #4a90e2;
                color: #4a90e2;
            }
            .sample-question-button button:hover {
                background-color: #4a90e2;
                color: white;
            }
            .stAlert {
                color: #f5f5f5 !important;
                background-color: #2e3b4e !important;
            }
            label[for="file_uploader"] {
                color: #f0f0f0 !important;
                font-weight: bold;
            }
        """
    else:
        return """
            html, body, .stApp {
                background: linear-gradient(to right, #eef2f3, #8e9eab);
                color: #1e1e2f;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .stApp {
                padding: 2rem 4rem;
            }
            h1, h2, h3, .stMarkdown, .stTextInput > label, .stFileUploader > label {
                font-weight: 600;
                color: #1a1a1a;
            }
            .stFileUploader {
                border: 2px dashed #2a9df4;
                padding: 1rem;
                background-color: #f9fbfd;
                border-radius: 10px;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: #dde5ed;
                border-radius: 8px 8px 0 0;
                font-size: 16px;
                font-weight: bold;
                color: #333;
                padding: 1rem;
                margin-right: 3px;
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                background-color: #2a9df4;
                color: white;
                font-weight: bold;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .stButton > button {
                background-color: #2a9df4;
                color: white;
                font-size: 16px;
                border-radius: 8px;
                padding: 10px 20px;
            }
            .stButton > button:hover {
                background-color: #1b7cc3;
            }
            .stTextInput input {
                padding: 0.75rem;
                border-radius: 6px;
                border: 1px solid #ccc;
            }
            .sample-question-button button {
                margin-right: 8px;
                margin-bottom: 10px;
                background-color: #ffffff;
                border: 2px solid #2a9df4;
                color: #2a9df4;
                border-radius: 5px;
                padding: 6px 16px;
            }
            .sample-question-button button:hover {
                background-color: #2a9df4;
                color: white;
            }
        """