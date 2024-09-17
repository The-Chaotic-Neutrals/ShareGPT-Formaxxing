import jsonlines
import os
import logging
import language_tool_python

class GrammarMaxxer:
    def __init__(self, input_file, toggles):
        self.input_file = input_file
        self.toggles = toggles
        self.tool = language_tool_python.LanguageTool('en-US')

    def validate_file(self):
        """Validate the selected file."""
        if not self.input_file.endswith('.jsonl'):
            logging.error("Invalid file type. Please select a .jsonl file.")
            return False
        return True

    def prepare_output_file(self):
        """Prepare the output file path."""
        output_dir = "corrected"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.input_file))[0]
        return os.path.join(output_dir, f"{base_name}-corrected.jsonl")

    def process_file(self, output_file, update_corrections_callback):
        """Process the input file and write the corrected text to the output file."""
        with jsonlines.open(self.input_file) as reader:
            with jsonlines.open(output_file, mode='w') as writer:
                for conversation in reader:
                    corrected_conversation = self.correct_conversation(conversation, update_corrections_callback)
                    writer.write(corrected_conversation)

    def correct_conversation(self, conversation, update_corrections_callback):
        """Correct the text in a conversation and update the live tracker."""
        for turn in conversation.get('conversations', []):
            if turn.get('from') == 'gpt':
                original_text = turn.get('value', '')
                corrected_text = self.correct_text(original_text)
                turn['value'] = corrected_text
                update_corrections_callback(original_text, corrected_text)
        return conversation

    def correct_text(self, text):
        """Correct text using a multi-step process."""
        corrections = {
            'grammar': self.correct_with_grammar
        }
        for key, func in corrections.items():
            if self.toggles[key].get() == 'on':
                text = func(text)
        return text.strip()

    def correct_with_grammar(self, text):
        """Correct grammar using LanguageTool."""
        matches = self.tool.check(text)
        corrected_text = language_tool_python.utils.correct(text, matches)
        return corrected_text
