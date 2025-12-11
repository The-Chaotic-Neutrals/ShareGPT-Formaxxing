# import io
import json
import os
import re
from wordcloud import WordCloud
from PIL import Image

class WordCloudGenerator:
    def __init__(self, theme, update_status_callback):
        self.theme = theme
        self.update_status = update_status_callback
    
    def generate_wordcloud(self, file_path, width=1600, height=1200):
        self.update_status("Loading file...")
        try:
            text = self.load_and_process_text(file_path)
            if not text.strip():
                self.update_status("No text found in the file to generate a word cloud.")
                return
            self.update_status("Generating word cloud...")
            wordcloud = WordCloud(width=width, height=height, background_color=self.theme['bg']).generate(text)
           
            # Get the word cloud as a PIL Image
            pil_image = wordcloud.to_image()
           
            # Save the image to file - default to outputs folder in repo root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(script_dir)
            output_dir = os.path.join(repo_root, "outputs", "wordclouds")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "wordcloud.png")
            pil_image.save(output_path)
           
            # Update GUI with the generated image and status
            self.update_status(f"Word cloud generated and saved to {output_path}.")
            return pil_image
           
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
    
    def load_and_process_text(self, file_path):
        text = ''
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    data = json.loads(line)
                    if 'conversations' in data:
                        for message in data['conversations']:
                            if message.get('from') != 'system' and 'value' in message:
                                value = re.sub(r'[^\w\s]', '', message['value'].lower())
                                text += ' ' + value
        except Exception as e:
            self.update_status(f"Error processing file: {str(e)}")
        return text