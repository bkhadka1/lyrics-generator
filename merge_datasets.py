"""
COMPLETE DATASET MERGER
Combines Musixmatch scraped data + Kaggle Spotify dataset
"""

import json
import pandas as pd
import kagglehub
from pathlib import Path
from tqdm import tqdm
import re
from collections import Counter

class DatasetMerger:
    def __init__(self, output_dir="data_merged"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.seen_lyrics = set()  # For deduplication
        self.seen_ids = set()
        
    def clean_lyrics(self, lyrics):
        """Clean and normalize lyrics"""
        if not lyrics or not isinstance(lyrics, str):
            return ""
        
        # Remove common spam patterns
        spam_patterns = [
            r'\[.*?\]',  # [Verse 1], [Chorus]
            r'http[s]?://\S+',  # URLs
            r'see.*lyrics.*at',
            r'visit.*for.*lyrics',
            r'you might also like',
            r'\d+ contributors?',
            r'embed',
        ]
        
        for pattern in spam_patterns:
            lyrics = re.sub(pattern, '', lyrics, flags=re.IGNORECASE)
        
        # Clean whitespace
        lyrics = re.sub(r'\n{3,}', '\n\n', lyrics)
        lyrics = re.sub(r' {2,}', ' ', lyrics)
        lyrics = lyrics.strip()
        
        return lyrics
    
    def is_quality_lyrics(self, lyrics, min_words=50, max_words=2000):
        """Check if lyrics meet quality standards"""
        if not lyrics:
            return False
        
        lyrics_lower = lyrics.lower()
        
        # Skip instrumentals
        if any(word in lyrics_lower for word in ['instrumental', 'no lyrics']):
            return False
        
        # Check word count
        words = lyrics.split()
        if len(words) < min_words or len(words) > max_words:
            return False
        
        # Check line count
        lines = [l.strip() for l in lyrics.split('\n') if l.strip()]
        if len(lines) < 10:
            return False
        
        # Check for spam (too many repeated words)
        word_counts = Counter(words)
        if word_counts.most_common(1)[0][1] > len(words) * 0.3:
            return False
        
        # Check unique words ratio
        if len(set(words)) < len(words) * 0.25:
            return False
        
        return True
    
    def create_lyrics_hash(self, lyrics):
        """Create hash for deduplication"""
        # Use first 100 words as fingerprint
        words = lyrics.lower().split()[:100]
        return ' '.join(sorted(set(words)))
    
    def process_musixmatch_data(self, musixmatch_file="data/lyrics_raw.jsonl"):
        """Process existing Musixmatch scraped data"""
        print("\n📂 Processing Musixmatch data...")
        
        if not Path(musixmatch_file).exists():
            print(f"   ⚠️  File not found: {musixmatch_file}")
            return []
        
        processed = []
        
        with open(musixmatch_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Musixmatch"):
                try:
                    entry = json.loads(line)
                    
                    lyrics = entry.get('lyrics', '')
                    lyrics_clean = self.clean_lyrics(lyrics)
                    
                    if not self.is_quality_lyrics(lyrics_clean):
                        continue
                    
                    # Deduplication
                    lyrics_hash = self.create_lyrics_hash(lyrics_clean)
                    if lyrics_hash in self.seen_lyrics:
                        continue
                    self.seen_lyrics.add(lyrics_hash)
                    
                    processed.append({
                        'source': 'musixmatch',
                        'track_id': entry.get('track_id', ''),
                        'track_name': entry.get('track_name', 'Unknown'),
                        'artist_name': entry.get('artist_name', 'Unknown'),
                        'album_name': entry.get('album_name', ''),
                        'lyrics': lyrics_clean,
                        'word_count': len(lyrics_clean.split())
                    })
                    
                except Exception as e:
                    continue
        
        print(f"   ✅ Processed {len(processed)} songs from Musixmatch")
        return processed
    
    def download_and_process_kaggle(self):
        """Download and process Kaggle Spotify dataset"""
        print("\n📥 Downloading Kaggle Spotify dataset...")
        
        try:
            # Download dataset
            path = kagglehub.dataset_download("notshrirang/spotify-million-song-dataset")
            print(f"   ✅ Downloaded to: {path}")
            
            # Find the CSV file
            dataset_path = Path(path)
            csv_files = list(dataset_path.glob("*.csv"))
            
            if not csv_files:
                print("   ❌ No CSV files found in dataset")
                return []
            
            csv_file = csv_files[0]
            print(f"   📂 Processing {csv_file.name}...")
            
            # Load CSV
            df = pd.read_csv(csv_file)
            print(f"   📊 Loaded {len(df)} rows")
            
            # Check available columns
            print(f"   Columns: {df.columns.tolist()}")
            
            processed = []
            
            # Process rows
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Kaggle"):
                try:
                    # Try different possible column names
                    lyrics = None
                    for col in ['text', 'lyrics', 'lyric', 'Lyrics', 'Text']:
                        if col in df.columns:
                            lyrics = row[col]
                            break
                    
                    if not lyrics or pd.isna(lyrics):
                        continue
                    
                    lyrics_clean = self.clean_lyrics(str(lyrics))
                    
                    if not self.is_quality_lyrics(lyrics_clean):
                        continue
                    
                    # Deduplication
                    lyrics_hash = self.create_lyrics_hash(lyrics_clean)
                    if lyrics_hash in self.seen_lyrics:
                        continue
                    self.seen_lyrics.add(lyrics_hash)
                    
                    # Extract metadata
                    track_name = row.get('song', row.get('track', row.get('title', 'Unknown')))
                    artist_name = row.get('artist', row.get('artist_name', 'Unknown'))
                    
                    processed.append({
                        'source': 'kaggle_spotify',
                        'track_id': f"kaggle_{idx}",
                        'track_name': str(track_name),
                        'artist_name': str(artist_name),
                        'album_name': '',
                        'lyrics': lyrics_clean,
                        'word_count': len(lyrics_clean.split())
                    })
                    
                except Exception as e:
                    continue
            
            print(f"   ✅ Processed {len(processed)} songs from Kaggle")
            return processed
            
        except Exception as e:
            print(f"   ❌ Error downloading Kaggle dataset: {e}")
            print("   💡 Make sure you have kagglehub installed: pip install kagglehub")
            return []
    
    def merge_and_save(self, musixmatch_data, kaggle_data):
        """Merge datasets and save in training format"""
        print("\n🔀 Merging datasets...")
        
        all_data = musixmatch_data + kaggle_data
        
        # Sort by artist to help with batching
        all_data.sort(key=lambda x: x['artist_name'])
        
        # Save raw merged data
        raw_file = self.output_dir / "lyrics_merged_raw.jsonl"
        with open(raw_file, 'w', encoding='utf-8') as f:
            for entry in all_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Save training format
        train_file = self.output_dir / "training_data_merged.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for entry in all_data:
                training_entry = {
                    "instruction": f"Write lyrics for a song titled '{entry['track_name']}' by {entry['artist_name']}",
                    "output": entry['lyrics']
                }
                f.write(json.dumps(training_entry, ensure_ascii=False) + '\n')
        
        # Create augmented version with varied prompts
        augmented_file = self.output_dir / "training_data_augmented.jsonl"
        self.create_augmented_prompts(all_data, augmented_file)
        
        print(f"\n✅ Merged dataset saved!")
        print(f"   📁 Raw data: {raw_file}")
        print(f"   📁 Training data: {train_file}")
        print(f"   📁 Augmented: {augmented_file}")
        
        return all_data
    
    def create_augmented_prompts(self, data, output_file):
        """Create varied instruction prompts"""
        
        prompt_templates = [
            "Write lyrics for a song titled '{title}' by {artist}",
            "Generate lyrics for '{title}' in the style of {artist}",
            "Create a song called '{title}' by {artist}",
            "Compose lyrics for {artist}'s song '{title}'",
            "Write a song titled '{title}' that sounds like {artist}",
        ]
        
        import random
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in data:
                # Original format
                original = {
                    "instruction": f"Write lyrics for a song titled '{entry['track_name']}' by {entry['artist_name']}",
                    "output": entry['lyrics']
                }
                f.write(json.dumps(original, ensure_ascii=False) + '\n')
                
                # 1 random variation
                template = random.choice(prompt_templates)
                augmented = {
                    "instruction": template.format(
                        title=entry['track_name'],
                        artist=entry['artist_name']
                    ),
                    "output": entry['lyrics']
                }
                f.write(json.dumps(augmented, ensure_ascii=False) + '\n')
    
    def analyze_merged_dataset(self, data):
        """Analyze the merged dataset"""
        print("\n📊 DATASET ANALYSIS")
        print("=" * 60)
        
        # Basic stats
        print(f"\nTotal songs: {len(data)}")
        
        # Source breakdown
        sources = Counter(entry['source'] for entry in data)
        print(f"\nBy source:")
        for source, count in sources.items():
            print(f"   {source}: {count} songs")
        
        # Word count stats
        word_counts = [entry['word_count'] for entry in data]
        print(f"\nLyrics length:")
        print(f"   Mean: {sum(word_counts)/len(word_counts):.0f} words")
        print(f"   Min: {min(word_counts)} words")
        print(f"   Max: {max(word_counts)} words")
        
        # Top artists
        artists = Counter(entry['artist_name'] for entry in data)
        print(f"\nTop 10 artists:")
        for artist, count in artists.most_common(10):
            print(f"   {artist}: {count} songs")
        
        # Quality distribution
        high_quality = sum(1 for entry in data if entry['word_count'] >= 150)
        print(f"\nQuality distribution:")
        print(f"   High quality (150+ words): {high_quality} ({high_quality/len(data)*100:.1f}%)")
        print(f"   Standard (50-149 words): {len(data) - high_quality}")


def run_complete_merge():
    """Run the complete merge pipeline"""
    
    print("=" * 60)
    print("🎵 DATASET MERGER: Musixmatch + Kaggle Spotify")
    print("=" * 60)
    
    merger = DatasetMerger(output_dir="data_merged")
    
    # Step 1: Process Musixmatch data
    musixmatch_data = merger.process_musixmatch_data("data/lyrics_raw.jsonl")
    
    # Step 2: Download and process Kaggle
    kaggle_data = merger.download_and_process_kaggle()
    
    # Step 3: Merge and save
    if not musixmatch_data and not kaggle_data:
        print("\n❌ No data to merge!")
        return
    
    merged_data = merger.merge_and_save(musixmatch_data, kaggle_data)
    
    # Step 4: Analyze
    merger.analyze_merged_dataset(merged_data)
    
    print("\n" + "=" * 60)
    print("✅ MERGE COMPLETE!")
    print("=" * 60)
    print(f"\n🎯 Total high-quality songs: {len(merged_data)}")
    print(f"\n📁 Use for training:")
    print(f"   data_merged/training_data_augmented.jsonl")
    print(f"\n🚀 Next step: Run train_model.py with this dataset")


if __name__ == "__main__":
    run_complete_merge()