"""
UPLOAD MODEL TO HUGGING FACE HUB
Solves storage issues and enables easy deployment
"""

from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path
import os

def upload_model_to_huggingface(
    model_path="lyrics_model/final_model",
    repo_name="lyrics-generator-gpt2",  # Change this to your desired name
    username=None,  # Your HuggingFace username
    private=False
):
    """
    Upload trained model to Hugging Face Hub
    
    Args:
        model_path: Local path to your trained model
        repo_name: Name for your HuggingFace repo
        username: Your HuggingFace username
        private: Whether to make repo private
    """
    
    print("=" * 70)
    print("🤗 UPLOADING MODEL TO HUGGING FACE HUB")
    print("=" * 70)
    
    # Check if model exists
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first using train_model.py")
        return None
    
    print(f"\n✅ Model found at {model_path}")
    
    # Get username if not provided
    if username is None:
        username = input("\nEnter your HuggingFace username: ").strip()
    
    # Create full repo ID
    repo_id = f"{username}/{repo_name}"
    
    print(f"\n📦 Repository: {repo_id}")
    print(f"🔒 Private: {private}")
    
    # Check if user is logged in
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"\n✅ Logged in as: {user_info['name']}")
    except Exception as e:
        print("\n❌ Not logged in to Hugging Face!")
        print("\n📝 To login, run:")
        print("   huggingface-cli login")
        print("\nOr use an access token:")
        print("   1. Go to https://huggingface.co/settings/tokens")
        print("   2. Create a token with 'write' access")
        print("   3. Run: huggingface-cli login")
        return None
    
    # Confirm upload
    print(f"\n⚠️  This will upload your model to: https://huggingface.co/{repo_id}")
    confirm = input("Continue? [y/n]: ").strip().lower()
    
    if confirm != 'y':
        print("❌ Upload cancelled")
        return None
    
    try:
        # Create repository
        print(f"\n📤 Creating repository...")
        create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✅ Repository created/verified")
        
        # Create model card
        print(f"\n📝 Creating model card...")
        create_model_card(model_path, repo_id)
        
        # Upload model files
        print(f"\n📤 Uploading model files...")
        print("   This may take a few minutes...")
        
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
        )
        
        print(f"\n✅ Upload complete!")
        print(f"🌐 Model available at: https://huggingface.co/{repo_id}")
        
        # Save repo ID for later use
        with open("huggingface_model_id.txt", "w") as f:
            f.write(repo_id)
        print(f"\n💾 Saved model ID to: huggingface_model_id.txt")
        
        return repo_id
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return None


def create_model_card(model_path, repo_id):
    """Create a README.md model card"""
    
    model_card = f"""---
language:
- en
license: mit
tags:
- text-generation
- lyrics
- music
- gpt2
- fine-tuned
datasets:
- custom
widget:
- text: "Write lyrics for a pop song about summer love"
---

# 🎵 AI Lyrics Generator

This model generates song lyrics based on prompts. It's a fine-tuned version of GPT-2 trained on a diverse dataset of song lyrics.

## Model Description

- **Base Model:** GPT-2
- **Task:** Lyrics Generation
- **Training Data:** Custom dataset of 10,000+ songs from various genres
- **Languages:** English

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}")

# Generate lyrics
prompt = "Write lyrics for a rock song about freedom"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs.input_ids,
    max_length=300,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)

lyrics = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(lyrics)
```

## Streamlit Demo

Try the interactive web app:
```bash
streamlit run app.py
```

## Training Details

- **Epochs:** 3
- **Batch Size:** 4
- **Learning Rate:** 5e-5
- **Training Time:** ~2 hours (GPU)

## Example Outputs

**Prompt:** "Write lyrics for a sad ballad about heartbreak"

**Output:**
```
I'm standing in the rain again
Thinking 'bout the way we were
Every memory cuts like glass
And I can't seem to let you go...
```

## Limitations

- May generate repetitive phrases
- Best with clear, specific prompts
- Quality varies by prompt complexity

## Citation

```
@misc{{lyrics-generator-gpt2,
  author = {{Your Name}},
  title = {{AI Lyrics Generator}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```

## License

MIT License
"""
    
    readme_path = Path(model_path) / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(model_card)
    
    print(f"   ✅ Model card created")


def quick_upload():
    """Interactive upload wizard"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🤗 HUGGING FACE MODEL UPLOAD WIZARD 🤗                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("\n📋 Prerequisites:")
    print("   1. HuggingFace account (free): https://huggingface.co/join")
    print("   2. Trained model at: lyrics_model/final_model/")
    print("   3. Logged in via: huggingface-cli login")
    
    print("\n" + "=" * 70)
    
    # Check login status
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"✅ Logged in as: {user['name']}")
        username = user['name']
    except:
        print("❌ Not logged in!")
        print("\n🔑 To login:")
        print("   1. Install: pip install huggingface_hub")
        print("   2. Run: huggingface-cli login")
        print("   3. Enter your token from: https://huggingface.co/settings/tokens")
        return
    
    # Model path
    print("\n📂 Model Configuration:")
    model_path = input("Model path [lyrics_model/final_model]: ").strip()
    if not model_path:
        model_path = "lyrics_model/final_model"
    
    # Repository name
    default_repo = "lyrics-generator-gpt2"
    repo_name = input(f"Repository name [{default_repo}]: ").strip()
    if not repo_name:
        repo_name = default_repo
    
    # Privacy
    private_choice = input("Make repository private? [y/n]: ").strip().lower()
    private = private_choice == 'y'
    
    # Upload
    print("\n" + "=" * 70)
    repo_id = upload_model_to_huggingface(
        model_path=model_path,
        repo_name=repo_name,
        username=username,
        private=private
    )
    
    if repo_id:
        print("\n" + "=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print(f"\n🌐 Your model: https://huggingface.co/{repo_id}")
        print(f"\n📝 Next steps:")
        print(f"   1. Update app.py to use: {repo_id}")
        print(f"   2. Deploy to Streamlit Cloud (no storage issues!)")
        print(f"   3. Share your model with the world! 🎉")


if __name__ == "__main__":
    # Check if huggingface_hub is installed
    try:
        import huggingface_hub
    except ImportError:
        print("❌ huggingface_hub not installed!")
        print("\nInstall with:")
        print("   pip install huggingface_hub")
        exit(1)
    
    quick_upload()