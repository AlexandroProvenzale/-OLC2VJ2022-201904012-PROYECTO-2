mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
[theme]

primaryColor=\"#6eb52f\"
backgroundColor=\"#f0f0f5\"
secondaryBackgroundColor=\"#e0e0ef\"
textColor=\"#262730\"
font="sans serif"
" > ~/.streamlit/config.toml