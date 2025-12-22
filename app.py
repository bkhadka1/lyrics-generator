"""
STREAMLIT LYRICS GENERATOR - WITH GENRE SELECTION
app.py - Enhanced web interface for lyrics generation
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="AI Lyrics Generator",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .lyrics-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        line-height: 1.8;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
    }
    .genre-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: #667eea;
        color: white;
        border-radius: 15px;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)"""
    
    model_sources = []
    
    # Option 1: Environment variable (best for deployment)
    hf_model_id = os.getenv("HUGGINGFACE_MODEL_ID")
    if hf_model_id:
        model_sources.append(("HuggingFace (env)", hf_model_id))
    
    # Option 2: Config file
    config_file = Path("huggingface_model_id.txt")
    if config_file.exists():
        with open(config_file, 'r') as f:
            model_id = f.read().strip()
            if model_id:
                model_sources.append(("HuggingFace (config)", model_id))
    
    # Option 3: Local model (development)
    local_path = "lyrics_model/final_model"
    if Path(local_path).exists():
        model_sources.append(("Local", local_path))
    
    # Try each source
    for source_name, model_path in model_sources:
        try:
            st.info(f"Loading model from {source_name}...")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            
            st.success(f"✅ Loaded from {source_name}")
            return tokenizer, model, device
            
        except Exception as e:
            st.warning(f"Could not load from {source_name}: {e}")
            continue
    
    # If all sources failed
    st.error("❌ Could not load model from any source!")
    st.markdown("""
    **Options:**
    1. **Train locally:** Run `python train_model.py`
    2. **Upload to HuggingFace:** Run `python upload_to_huggingface.py`
    3. **Set model ID:** Create `huggingface_model_id.txt` with your model ID
    4. **Environment variable:** Set `HUGGINGFACE_MODEL_ID=username/repo-name`
    """)
    st.stop()


def generate_lyrics(prompt, tokenizer, model, device, 
                   max_length=300, temperature=0.8, top_p=0.9):
    """Generate lyrics from prompt"""
    
    # Format prompt
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    
    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response
    if "### Response:" in generated:
        lyrics = generated.split("### Response:")[1].strip()
    else:
        lyrics = generated
    
    return lyrics


def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 AI Lyrics Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Create original song lyrics powered by AI</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading AI model..."):
        tokenizer, model, device = load_model()
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Generation Settings")
        
        temperature = st.slider(
            "Creativity (Temperature)",
            min_value=0.5,
            max_value=1.5,
            value=0.8,
            step=0.1,
            help="Higher = more creative but less coherent"
        )
        
        top_p = st.slider(
            "Diversity (Top-p)",
            min_value=0.5,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Controls randomness in word selection"
        )
        
        max_length = st.slider(
            "Max Length",
            min_value=100,
            max_value=500,
            value=300,
            step=50,
            help="Maximum length of generated lyrics"
        )
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Select a genre for better results
        - Be specific about theme and mood
        - Experiment with creativity settings
        - Try different templates
        - Add artist style for specific vibes
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info(f"""
        **Model:** GPT-2 Fine-tuned
        **Device:** {device.upper()}
        **Training Data:** 20,000+ songs
        """)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Your Prompt")
        
        # Genre selection
        st.subheader("🎸 Genre")
        genres = {
            "Any Genre": "",
            "Pop": "pop",
            "Rock": "rock",
            "Hip-Hop/Rap": "hip hop",
            "Country": "country",
            "R&B/Soul": "R&B",
            "Jazz": "jazz",
            "Blues": "blues",
            "Electronic/EDM": "electronic",
            "Indie": "indie",
            "Folk": "folk",
            "Metal": "metal",
            "Punk": "punk",
            "Reggae": "reggae",
            "Alternative": "alternative",
            "Singer-Songwriter": "singer-songwriter"
        }
        
        col_genre1, col_genre2 = st.columns([2, 1])
        with col_genre1:
            selected_genre = st.selectbox(
                "Select music genre:",
                list(genres.keys()),
                help="Choose a music genre for your lyrics",
                label_visibility="collapsed"
            )
        
        # Templates
        st.subheader("📋 Template")
        templates = {
            "Custom": "",
            "Love Song": "Write lyrics for a {genre} song about falling in love",
            "Breakup Song": "Write lyrics for a {genre} ballad about heartbreak and moving on",
            "Party Anthem": "Write lyrics for an upbeat {genre} song about having a good time and celebrating",
            "Inspirational": "Write lyrics for a motivational {genre} song about overcoming challenges",
            "Summer Vibes": "Write lyrics for a {genre} song about summer, sunshine, and good times",
            "Nostalgia": "Write lyrics for a {genre} song about memories and the past",
            "Road Trip": "Write lyrics for a {genre} song about adventure and traveling",
            "Friendship": "Write lyrics for a {genre} song about friendship and loyalty",
            "Dreams & Ambitions": "Write lyrics for a {genre} song about chasing dreams",
            "City Life": "Write lyrics for a {genre} song about urban life and city experiences"
        }
        
        template = st.selectbox(
            "Choose a template or write custom:",
            list(templates.keys()),
            help="Select a template or choose Custom to write your own"
        )
        
        # Format template with genre
        if template != "Custom":
            genre_text = genres[selected_genre] if genres[selected_genre] else ""
            if genre_text:
                default_prompt = templates[template].format(genre=genre_text)
            else:
                # Remove {genre} and clean up
                default_prompt = templates[template].replace("{genre} ", "").replace(" {genre}", "")
        else:
            default_prompt = ""
        
        # Main prompt area
        st.subheader("✍️ Prompt")
        prompt = st.text_area(
            "Describe your song:",
            value=default_prompt,
            height=120,
            placeholder="E.g., Write lyrics for a country song about missing home...",
            label_visibility="collapsed"
        )
        
        # Auto-add genre to custom prompts if not mentioned
        final_prompt = prompt
        if genres[selected_genre] and template == "Custom":
            if genres[selected_genre].lower() not in prompt.lower() and prompt.strip():
                final_prompt = f"Write lyrics for a {genres[selected_genre]} song: {prompt}"
        
        # Advanced options
        with st.expander("🎯 Advanced Options"):
            col_adv1, col_adv2 = st.columns(2)
            
            with col_adv1:
                include_artist = st.text_input(
                    "Artist style:",
                    placeholder="e.g., Taylor Swift",
                    help="Generate lyrics in the style of a specific artist"
                )
                mood = st.selectbox(
                    "Mood:",
                    ["Any", "Happy", "Sad", "Energetic", "Chill", "Angry", "Romantic", "Melancholic", "Hopeful"]
                )
            
            with col_adv2:
                include_theme = st.text_input(
                    "Specific theme:",
                    placeholder="e.g., nature, freedom",
                    help="Add a specific theme or topic"
                )
                song_structure = st.checkbox(
                    "Include structure markers",
                    help="Add Verse, Chorus, Bridge markers"
                )
            
            # Modify prompt based on advanced options
            if include_artist:
                final_prompt += f" in the style of {include_artist}"
            if mood != "Any":
                final_prompt += f" with a {mood.lower()} mood"
            if include_theme:
                final_prompt += f" about {include_theme}"
            if song_structure:
                final_prompt += ". Include clear verse and chorus markers like [Verse 1], [Chorus], [Bridge]."
        
        # Show final prompt preview
        with st.expander("👁️ Preview Final Prompt"):
            st.code(final_prompt, language=None)
        
        generate_button = st.button("🎵 Generate Lyrics", type="primary", use_container_width=True)
    
    with col2:
        st.header("🎤 Generated Lyrics")
        
        if generate_button:
            if not final_prompt.strip():
                st.error("Please enter a prompt!")
            else:
                # Show selected genre badge
                if genres[selected_genre]:
                    st.markdown(f'<div class="genre-badge">🎸 {selected_genre}</div>', unsafe_allow_html=True)
                
                with st.spinner("✨ Creating your lyrics..."):
                    # Progress bar for effect
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Generate lyrics
                    lyrics = generate_lyrics(
                        prompt=final_prompt,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p
                    )
                    
                    progress_bar.empty()
                
                # Display results
                st.markdown('<div class="lyrics-box">', unsafe_allow_html=True)
                st.markdown(lyrics)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Action buttons
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.download_button(
                        "💾 Download",
                        lyrics,
                        file_name=f"lyrics_{selected_genre.lower().replace('/', '_')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col_b:
                    if st.button("🔄 Regenerate", use_container_width=True):
                        st.rerun()
                with col_c:
                    if st.button("📋 Copy", use_container_width=True, help="Click to show copyable text"):
                        st.code(lyrics, language=None)
                
                # Metadata
                st.caption(f"Generated with: Temperature={temperature}, Top-p={top_p}, Max Length={max_length}")
        
        else:
            st.info("👈 Select a genre, choose a template or write custom prompt, then click 'Generate Lyrics' to start!")
            
            # Example showcase
            st.markdown("### 🎯 Example Output")
            st.markdown("""
            ```
            [Verse 1]
            Walking down these empty streets tonight
            Thinking 'bout the way you held me tight
            Every corner brings back memories
            Of you and me beneath the willow trees
            
            [Chorus]
            I'm missing you, can't you see
            Every moment, you're all I need
            Come back to me, where you belong
            You're the melody to my song
            ```
            """)
    
    # Footer
    st.markdown("---")
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    
    with col_info1:
        st.metric("Model", "GPT-2 Fine-tuned")
    with col_info2:
        st.metric("Device", device.upper())
    with col_info3:
        st.metric("Genres", "15+")
    with col_info4:
        st.metric("Status", "Ready ✅")
    
    # Examples section
    with st.expander("📚 Tips for Great Lyrics"):
        st.markdown("""
        ### 🎵 Genre-Specific Tips:
        
        **Pop:** Focus on catchy hooks, relatable emotions, upbeat or emotional themes
        **Rock:** Emphasize energy, rebellion, powerful emotions, strong imagery
        **Hip-Hop:** Include wordplay, rhythm, storytelling, cultural references
        **Country:** Tell stories, use vivid imagery, focus on everyday life and emotions
        **R&B/Soul:** Smooth flow, emotional depth, romantic or introspective themes
        **Electronic:** Repetitive hooks, energy, club vibes, minimalist lyrics
        
        ### ✍️ General Tips:
        - Be specific: "summer beach party" vs "having fun"
        - Include emotions: joy, heartbreak, excitement, nostalgia
        - Mention setting: city, countryside, night, day
        - Add sensory details: sounds, sights, feelings
        - Try artist styles: "in the style of [artist]"
        - Use mood selectors for better tone matching
        
        ### 🎨 Creative Techniques:
        - Combine genres: "Write electronic pop lyrics"
        - Mix moods: "energetic but melancholic"
        - Add constraints: "with ocean metaphors"
        - Request structure: "with a strong chorus"
        """)


if __name__ == "__main__":
    main()