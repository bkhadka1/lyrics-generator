# 🎵 AI Lyrics Generator

An AI-powered web app that generates original song lyrics using a fine-tuned GPT-2 model.

## 🚀 Live Demo

[Try it here!](https://bkhadka-lyrics-gen.streamlit.app/)

## Features

- Generate lyrics in various genres and styles
- Adjustable creativity settings
- Download generated lyrics
- Powered by GPT-2 fine-tuned on 20,000+ songs

## Setup

1. Clone repository:
   \`\`\`bash
   git clone https://github.com/YOUR_USERNAME/lyrics-generator.git
   cd lyrics-generator
   \`\`\`

2. Create virtual environment:
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   \`\`\`

3. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. Configure model:
   - Create \`huggingface_model_id.txt\`
   - Add your HuggingFace model ID

5. Run app:
   \`\`\`bash
   streamlit run app.py
   \`\`\`

## Model

Trained model available on [HuggingFace](YOUR_HF_MODEL_URL)

## Tech Stack

- **Model:** GPT-2 (Transformers)
- **Frontend:** Streamlit
- **Storage:** HuggingFace Hub
- **Deployment:** Streamlit Cloud

## License

MIT License
