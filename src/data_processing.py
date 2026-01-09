"""
Data Processing Module
Cleans and normalizes product catalogue data
"""

import pandas as pd
import re
import ast
from typing import List, Dict
import json


class DataProcessor:
    """Handles data cleaning and normalization for product catalogue"""
    
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV"""
        print(f"Loading data from {self.input_path}...")
        self.df = pd.read_csv(self.input_path)
        print(f"Loaded {len(self.df)} records")
        return self.df
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text:
        - Remove extra whitespace
        - Lowercase
        - Remove special characters (keep alphanumeric and spaces)
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep letters, numbers, spaces, and hyphens
        text = re.sub(r'[^a-zA-Z0-9\s-]', '', text)
        
        # Lowercase
        text = text.lower().strip()
        
        return text
    
    def parse_tags(self, tags_str: str) -> List[str]:
        """Parse tags from string representation of list"""
        if pd.isna(tags_str) or tags_str == '[]':
            return []
        
        try:
            # Try to parse as Python list
            tags = ast.literal_eval(tags_str)
            return [self.clean_text(tag) for tag in tags if tag]
        except:
            return []
    
    def create_searchable_text(self, row: pd.Series) -> str:
        """
        Combine relevant fields into one searchable text field
        This is what we'll use for embedding and matching
        """
        components = []
        
        # Add title (most important)
        if pd.notna(row['title']):
            components.append(str(row['title']))
        
        # Add vendor/brand
        if pd.notna(row['vendor']):
            components.append(str(row['vendor']))
        
        # Add category
        if pd.notna(row['category']):
            components.append(str(row['category']))
        
        # Add important tags (filter out generic ones)
        tags = self.parse_tags(row['tags'])
        # Filter out very generic or noise tags
        filtered_tags = [t for t in tags if len(t) > 2 and t not in ['brand', 'type', 'category']]
        if filtered_tags:
            components.append(' '.join(filtered_tags[:5]))  # Limit to 5 tags
        
        # Combine all components
        full_text = ' '.join(components)
        
        return full_text
    
    def process(self) -> pd.DataFrame:
        """
        Main processing pipeline:
        1. Load data
        2. Clean fields
        3. Create searchable text
        4. Remove duplicates
        """
        # Load data
        self.load_data()
        
        print("Cleaning data...")
        
        # Parse tags
        self.df['tags_parsed'] = self.df['tags'].apply(self.parse_tags)
        
        # Create searchable text (original, not cleaned for embedding)
        self.df['searchable_text'] = self.df.apply(self.create_searchable_text, axis=1)
        
        # Create cleaned version for display
        self.df['title_cleaned'] = self.df['title'].apply(self.clean_text)
        self.df['vendor_cleaned'] = self.df['vendor'].apply(self.clean_text)
        
        # Remove rows with empty searchable text
        initial_count = len(self.df)
        self.df = self.df[self.df['searchable_text'].str.strip() != '']
        removed = initial_count - len(self.df)
        if removed > 0:
            print(f"Removed {removed} records with empty searchable text")
        
        # Remove duplicates based on searchable text
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['searchable_text'], keep='first')
        duplicates = initial_count - len(self.df)
        if duplicates > 0:
            print(f"Removed {duplicates} duplicate records")
        
        # Reset index
        self.df = self.df.reset_index(drop=True)
        
        print(f"Processing complete. Final dataset: {len(self.df)} records")
        
        return self.df
    
    def save_processed(self, output_path: str):
        """Save processed data"""
        if self.df is None:
            raise ValueError("No data to save. Run process() first.")
        
        self.df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
        
        # Save statistics
        stats = {
            'total_records': len(self.df),
            'unique_vendors': self.df['vendor'].nunique(),
            'unique_categories': self.df['category'].nunique(),
            'avg_text_length': self.df['searchable_text'].str.len().mean(),
            'sample_records': self.df[['product_id', 'title', 'searchable_text']].head(5).to_dict('records')
        }
        
        stats_path = output_path.replace('.csv', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to {stats_path}")


if __name__ == "__main__":
    # Process data
    processor = DataProcessor('data/product_catalogue.csv')
    df_processed = processor.process()
    processor.save_processed('data/product_catalogue_processed.csv')
    
    print("\n✓ Data processing complete!")