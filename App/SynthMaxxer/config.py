# Endpoint Fields
INFERENCE_API_ENDPOINT = "CHOSEN ENDPOINT HERE" + "/v1/messages" # v1/messages is for Claude endpoint, edit yours accordingly.
INFERENCE_API_KEY = "YOUR API KEY HERE"
MODEL = "Place the name of your model here"

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": INFERENCE_API_KEY,
    # Anthropic Specific Headers
    "anthropic-version": "2023-06-01"
}

# Baseline Starting Message
BASELINE_CLAUDE_CONFIG = {
    "DIRECTORY_NAME": "uncurated_raw_claude_gens",
    "ASSISTANT_START_TAG": "<claude_turn>",
    "ASSISTANT_END_TAG": "</claude_turn>",
    "USER_START_TAG": "<human_turn>",
    "USER_END_TAG": "</human_turn>",
    "USER_FIRST_MESSAGE": "You are an innovative AI assistant named Claude created by Anthropic to be helpful, harmless, and honest.",
    "ASSISTANT_FIRST_MESSAGE": "Hello! I am Claude, an innovative AI Assistant developed by Anthropic. How can I assist you today?",
    "SYSTEM_MESSAGE": "The following is a conversation between Claude, innovative AI Assistant developed by Anthropic, and a user of Claude.",
    "IsInstruct": True
}

ACTIVE_CONFIG = BASELINE_CLAUDE_CONFIG

REFUSAL_PHRASES = [
    "Upon further reflection",
    "I can't engage"
]

FORCE_RETRY_PHRASES = [
    "shivers down"
]

