"""
DEPLOYMENT CONFIGURATION FILES
For Streamlit Cloud, Hugging Face Spaces, etc.
"""

# ============================================
# requirements.txt
# ============================================

requirements_txt = """# requirements.txt - For Streamlit Cloud deployment

torch>=2.0.0
transformers>=4.30.0
streamlit>=1.25.0
huggingface_hub>=0.16.0
"""

# ============================================
# .streamlit/config.toml
# ============================================

streamlit_config = """[theme]
primaryColor = "#667eea"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
"""

# ============================================
# README.md for GitHub
# ============================================

github_readme = """# 🎵 AI Lyrics Generator

An AI-powered web app that generates original song lyrics using a fine-tuned GPT-2 model.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL_HERE)

## Features

- 🎵 Generate lyrics in various genres and styles
- 🎨 Adjustable creativity and length settings
- 💾 Download generated lyrics
- 🎯 Template prompts for quick start
- ⚡ Fast generation (powered by GPT-2)

## Quick Start

### Option 1: Use Online (Recommended)
Visit [YOUR_STREAMLIT_URL_HERE](YOUR_STREAMLIT_URL_HERE)

### Option 2: Run Locally

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/lyrics-generator.git
cd lyrics-generator

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

## How It Works

1. **Model:** Fine-tuned GPT-2 on 20,000+ song lyrics
2. **Training:** Custom dataset from multiple sources
3. **Deployment:** Streamlit Cloud + HuggingFace Hub
4. **No Storage Issues:** Model loaded from HuggingFace

## Model

The model is hosted on HuggingFace: [YOUR_HF_MODEL_ID](https://huggingface.co/YOUR_HF_MODEL_ID)

## Configuration

Set the HuggingFace model ID in one of these ways:

1. **Environment Variable:**
   ```bash
   export HUGGINGFACE_MODEL_ID="username/lyrics-generator-gpt2"
   ```

2. **Config File:**
   Create `huggingface_model_id.txt`:
   ```
   username/lyrics-generator-gpt2
   ```

3. **Streamlit Secrets:**
   In `.streamlit/secrets.toml`:
   ```toml
   HUGGINGFACE_MODEL_ID = "username/lyrics-generator-gpt2"
   ```

## Training Your Own Model

```bash
# 1. Collect data
python merge_datasets.py

# 2. Train model
python train_model.py

# 3. Upload to HuggingFace
python upload_to_huggingface.py

# 4. Update model ID in config
```

## Tech Stack

- **Frontend:** Streamlit
- **Model:** GPT-2 (Transformers)
- **Storage:** HuggingFace Hub
- **Deployment:** Streamlit Cloud

## Examples

**Prompt:** "Write lyrics for a pop song about summer love"

**Output:**
```
Under the golden sun, we dance until the day is done
Your smile lights up the sky, like fireworks in July
We're young and free, just you and me
This summer love is all we need...
```

## License

MIT License - Feel free to use and modify!

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## Credits

Built with ❤️ using Streamlit and Transformers
"""

# ============================================
# .streamlit/secrets.toml (template)
# ============================================

secrets_template = """# .streamlit/secrets.toml
# Copy this to .streamlit/secrets.toml and fill in your values
# DO NOT commit this file to GitHub!

# HuggingFace Model ID
HUGGINGFACE_MODEL_ID = "your-username/lyrics-generator-gpt2"

# Optional: HuggingFace token for private models
# HUGGINGFACE_TOKEN = "hf_xxxxxxxxxxxxx"
"""

# ============================================
# .gitignore
# ============================================

gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/

# Data
data/
data_merged/
data_cleaned/
*.jsonl
*.csv

# Model (don't commit local model)
lyrics_model/

# Secrets
.streamlit/secrets.toml
huggingface_token.txt

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
"""

# ============================================
# Save all files
# ============================================

def create_deployment_files():
    """Create all necessary deployment configuration files"""
    
    print("📝 Creating deployment configuration files...\n")
    
    files = {
        "requirements.txt": requirements_txt,
        ".streamlit/config.toml": streamlit_config,
        ".streamlit/secrets.toml.example": secrets_template,
        "README.md": github_readme,
        ".gitignore": gitignore,
    }
    
    for filepath, content in files.items():
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"   ✅ Created: {filepath}")
    
    # Create huggingface_model_id.txt template
    hf_id_file = Path("huggingface_model_id.txt.example")
    with open(hf_id_file, 'w') as f:
        f.write("your-username/lyrics-generator-gpt2\n")
    print(f"   ✅ Created: {hf_id_file}")
    
    print("\n✅ All deployment files created!")
    print("\n📋 Next steps:")
    print("   1. Upload model: python upload_to_huggingface.py")
    print("   2. Copy secrets template: cp .streamlit/secrets.toml.example .streamlit/secrets.toml")
    print("   3. Edit secrets.toml with your HuggingFace model ID")
    print("   4. Test locally: streamlit run app.py")
    print("   5. Deploy to Streamlit Cloud!")


# ============================================
# Streamlit Cloud Deployment Guide
# ============================================

def print_deployment_guide():
    """Print step-by-step deployment guide"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║               🚀 STREAMLIT CLOUD DEPLOYMENT GUIDE 🚀                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

📋 PREREQUISITES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ✅ Trained model
2. ✅ GitHub account
3. ✅ HuggingFace account
4. ✅ Streamlit Cloud account (free)


STEP 1: UPLOAD MODEL TO HUGGINGFACE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

$ python upload_to_huggingface.py

This uploads your model to HuggingFace Hub.
Example: https://huggingface.co/username/lyrics-generator-gpt2

✅ Model is now hosted (no GitHub storage issues!)


STEP 2: PREPARE GITHUB REPOSITORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

$ git init
$ git add app.py requirements.txt README.md .streamlit/config.toml
$ git commit -m "Initial commit"
$ git remote add origin https://github.com/YOUR_USERNAME/lyrics-generator.git
$ git push -u origin main

⚠️  DO NOT commit:
   - lyrics_model/ (model files)
   - data/ (training data)
   - .streamlit/secrets.toml (secrets)

✅ .gitignore already excludes these!


STEP 3: DEPLOY TO STREAMLIT CLOUD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Go to: https://share.streamlit.io
2. Click "New app"
3. Connect your GitHub repository
4. Select:
   - Repository: YOUR_USERNAME/lyrics-generator
   - Branch: main
   - Main file: app.py
5. Click "Advanced settings"
6. Add secrets:
   ```
   HUGGINGFACE_MODEL_ID = "username/lyrics-generator-gpt2"
   ```
7. Click "Deploy"!

✅ Your app will be live at: https://YOUR_APP.streamlit.app


STEP 4: UPDATE README (Optional)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Replace placeholders in README.md:
- YOUR_STREAMLIT_URL_HERE → your actual URL
- YOUR_HF_MODEL_ID → your HuggingFace model ID
- YOUR_USERNAME → your GitHub username


TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ "Model not found"
   → Check HUGGINGFACE_MODEL_ID in Streamlit secrets
   → Make sure model is public on HuggingFace

❌ "Out of memory"
   → Use distilgpt2 instead of gpt2-medium
   → Streamlit Cloud has 1GB RAM limit

❌ "App is slow"
   → Model loads on first request (takes ~30s)
   → Subsequent requests are fast (cached)


BENEFITS OF THIS APPROACH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ No GitHub storage issues (model on HuggingFace)
✅ Easy updates (just re-upload model)
✅ Shareable model (others can use it too)
✅ Free hosting (Streamlit Cloud + HuggingFace)
✅ Professional deployment


🎉 DONE! Your app is live and accessible to anyone!

Share it: https://YOUR_APP.streamlit.app
    """)


if __name__ == "__main__":
    create_deployment_files()
    print("\n" + "=" * 70)
    print_deployment_guide()