import io
import json
import os
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
from PIL import ImageTk

class WordCloudGenerator:
    def __init__(self, theme, update_status_callback):
        self.theme = theme
        self.update_status = update_status_callback

    def generate_wordcloud(self, file_path):
        self.update_status("Loading file...")
        try:
            text = self.load_and_process_text(file_path)
            if not text.strip():
                self.update_status("No text found in the file to generate a word cloud.")
                return

            self.update_status("Generating word cloud...")
            wordcloud = WordCloud(width=616, height=308, background_color='#2e2e2e').generate(text)
            
            # Save the word cloud to an image stream
            image_stream = io.BytesIO()
            self.save_wordcloud_image(wordcloud, image_stream)
            
            # Save the image to file
            output_dir = "wordclouds"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "wordcloud.png")
            image_stream.seek(0)
            img = Image.open(image_stream)
            img.save(output_path)
            
            # Convert image to Tkinter format
            img_tk = ImageTk.PhotoImage(img)

            # Update GUI with the generated image and status
            self.update_status(f"Word cloud generated and saved to {output_path}.")
            return img_tk
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")

    def save_wordcloud_image(self, wordcloud, image_stream):
        """ Save the word cloud image to the given stream """
        # Ensure figure creation and saving happens in the main thread
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(image_stream, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()

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
