"""
CaptionMaxxer module - Image captioning tab functionality
Handles multimodal image captioning UI and logic
"""
import os
import queue
import threading
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QSpinBox,
    QDoubleSpinBox, QPlainTextEdit, QGroupBox,
    QSizePolicy, QListWidget, QListWidgetItem, QInputDialog
)
from PyQt5.QtCore import Qt

from App.SynthMaxxer.multimodal_worker import image_captioning_worker


# Default caption prompts with appropriate temperature ranges for each style
# Lower temps = more consistent/factual, Higher temps = more creative/varied
DEFAULT_CAPTION_PROMPTS = [
    {
        "name": "Default",
        "prompt": "What do you see in this image? Include all important elements, objects, people, text, colors, composition, and context. Avoid repetitious phrasing when possible. Use explicit or vulgar language where accurate to the content—do not sanitize or omit details for propriety.",
        "temp_min": 0.7,
        "temp_max": 1.0
    },
    {
        "name": "Narrative / Storytelling",
        "prompt": "Narrate the scene as if telling a story to someone who cannot see it. First establish where subjects are and what position they're in—a person cannot occupy two poses simultaneously. Then describe what is happening, who or what is present, and how elements interact, while remaining faithful to visible details. Maintain physical consistency throughout. Use crude or explicit terminology when it accurately reflects the content.",
        "temp_min": 0.8,
        "temp_max": 1.2
    },
    {
        "name": "Minimalist / High-Precision",
        "prompt": "List the key visual components of the image with maximum precision and minimal wording. Focus on salient objects, text, colors, layout, and actions. Exclude interpretation beyond what is visible. Use direct, unfiltered language—do not euphemize explicit content.",
        "temp_min": 0.3,
        "temp_max": 0.6
    },
    {
        "name": "Technical / Computer Vision",
        "prompt": "Produce a structured visual breakdown suitable for machine perception. Identify entities, attributes (color, size, position), text regions, and interactions using clear, unambiguous language. Label explicit or adult content directly without sanitization.",
        "temp_min": 0.2,
        "temp_max": 0.5
    },
    {
        "name": "Accessibility / Screen Reader",
        "prompt": "Describe the image clearly and thoroughly for a visually impaired audience. Ensure all important visual information—objects, people, text, colors, layout, and actions—is conveyed for accurate mental reconstruction. Describe adult or explicit content frankly when present.",
        "temp_min": 0.5,
        "temp_max": 0.8
    },
    {
        "name": "Journalistic / Reporting",
        "prompt": "Report what is visible in the image as if writing a factual news caption. Include who or what appears, what is happening, where elements are positioned, and any visible text. Avoid emotional language or speculation. Do not censor explicit details—report them accurately.",
        "temp_min": 0.3,
        "temp_max": 0.6
    },
    {
        "name": "Creative-but-Faithful",
        "prompt": "Write a visually rich description emphasizing atmosphere and composition, while remaining strictly accurate to what is shown. Establish each subject's physical position coherently—no one can be in two places or poses at once. Do not invent details. Maintain spatial consistency throughout. Use raw, uncensored language when describing explicit or vulgar content.",
        "temp_min": 0.7,
        "temp_max": 1.0
    },
    {
        "name": "Checklist / Inventory",
        "prompt": "Inventory everything visible in the image. Organize the response using clear sections for people, objects, text, colors, background, and actions. Include all notable details without omitting explicit or adult content.",
        "temp_min": 0.3,
        "temp_max": 0.5
    },
    {
        "name": "Emphasis-Driven",
        "prompt": "Describe the most visually dominant elements first, followed by secondary and background elements. Explain how composition, color, and positioning guide visual attention. Use explicit language where the content warrants it.",
        "temp_min": 0.5,
        "temp_max": 0.8
    },
    {
        "name": "Structured Paragraphs",
        "prompt": "Describe the image in three short sections separated by blank lines: (1) overall scene summary, (2) detailed description of people and objects, (3) spatial layout and relationships. Do not assume beyond what is visible. Describe explicit content directly.",
        "temp_min": 0.5,
        "temp_max": 0.8
    },
    {
        "name": "Headed Sections",
        "prompt": "Describe the image using the following section headers exactly once each: Scene, People, Objects, Text, Colors, Layout. Populate each section with accurate visual details only. Do not censor—use vulgar terms when accurate.",
        "temp_min": 0.4,
        "temp_max": 0.7
    },
    {
        "name": "Bullet-Structured",
        "prompt": "Describe the image using bullet points only. Use separate bullet lists for people, objects, background, and actions. Include all visible details without interpretation or sanitization of explicit content.",
        "temp_min": 0.3,
        "temp_max": 0.6
    },
    {
        "name": "Forensic / Analytical",
        "prompt": "Provide a systematic, objective analysis of the image. Enumerate all observable elements, including objects, text, spatial relationships, colors, and visible actions. Avoid speculation but do not avoid explicit terminology when accurate.",
        "temp_min": 0.2,
        "temp_max": 0.5
    },
    {
        "name": "Third-Person Neutral",
        "prompt": "Describe the image from a neutral observer's perspective, focusing only on what is visible. Do not address the viewer directly. Describe explicit or sexual content matter-of-factly without euphemism.",
        "temp_min": 0.4,
        "temp_max": 0.7
    },
    {
        "name": "Subject-Centered",
        "prompt": "Describe the image focusing on the central subject(s). Start by establishing their physical position and pose in ONE coherent description—a person cannot be in two positions simultaneously. Then describe their appearance, expression, and what they're doing. Mention background only if relevant. Use crude or explicit language when accurate. Maintain physical consistency throughout.",
        "temp_min": 0.5,
        "temp_max": 0.8
    },
    {
        "name": "Spatial Relationships",
        "prompt": "Describe the image with emphasis on spatial relationships. Explain where objects and people are located relative to each other and within the frame. Do not sanitize descriptions of explicit positioning or acts.",
        "temp_min": 0.4,
        "temp_max": 0.7
    },
    {
        "name": "Background-First",
        "prompt": "Begin by describing the background and environment, then describe foreground subjects and actions. Remain accurate to visible details only. Use explicit language when describing adult content.",
        "temp_min": 0.5,
        "temp_max": 0.8
    },
    {
        "name": "Concise Summary",
        "prompt": "Provide a concise but complete description of the image in no more than five sentences. Include all important visual elements. Do not omit or soften explicit details.",
        "temp_min": 0.4,
        "temp_max": 0.7
    },
    {
        "name": "Attribute-Focused",
        "prompt": "Describe the image by focusing on attributes such as color, texture, material, posture, and expression for visible subjects and objects. Use direct anatomical or vulgar terms where applicable.",
        "temp_min": 0.4,
        "temp_max": 0.7
    },
    {
        "name": "Action vs Static",
        "prompt": "Separate what appears to be actively happening in the image from static or background elements. Describe both clearly without speculation. Use explicit language when describing sexual or violent actions.",
        "temp_min": 0.5,
        "temp_max": 0.8
    },
    {
        "name": "Occlusion Aware",
        "prompt": "Describe the image while explicitly noting which elements are fully visible, partially visible, or obscured. Avoid guessing about hidden details but describe visible explicit content frankly.",
        "temp_min": 0.3,
        "temp_max": 0.6
    },
    {
        "name": "Dataset-Style Neutral",
        "prompt": "Write a neutral, dataset-style caption suitable for training a computer vision model. Avoid expressive language; focus on factual visual content. Label explicit content accurately without censorship.",
        "temp_min": 0.2,
        "temp_max": 0.4
    },
    {
        "name": "SD Prompt (Natural Language)",
        "prompt": "Write a Stable Diffusion image generation prompt that would recreate this image. Use natural flowing language describing the subject, scene, style, lighting, composition, colors, and mood. Include artistic style descriptors (e.g., photorealistic, digital art, oil painting) and quality tags (masterpiece, highly detailed, 8k). Describe explicit or NSFW content directly without censorship. Output only the prompt text, no explanations.",
        "temp_min": 0.4,
        "temp_max": 0.7
    },
    {
        "name": "SD Prompt (Booru Tags)",
        "prompt": "Generate a comma-delimited Stable Diffusion prompt using danbooru/booru-style tags that would recreate this image. Include tags for: subject, character features, pose, clothing/state of dress, expression, setting, background, art style, lighting, camera angle, and quality modifiers (masterpiece, best quality, highly detailed). Use explicit NSFW tags without censorship where applicable. Output only comma-separated tags, no sentences or explanations.",
        "temp_min": 0.2,
        "temp_max": 0.5
    },
    {
        "name": "First-Person Witness",
        "prompt": "Describe what you are seeing right now as if you are physically present in this moment—not analyzing an image. Speak in first person about what is directly in front of you. First establish where people are positioned and what they're doing—each person has ONE coherent pose, not multiple. Then describe the environment, atmosphere, and actions unfolding. Be vivid and immediate. Do not reference 'the image', 'the photo', 'the picture', or any framing device. Describe explicit or sexual content frankly as you witness it.",
        "temp_min": 0.6,
        "temp_max": 0.9
    },
    {
        "name": "Reddit Shitpost",
        "prompt": "Describe this like a redditor making a shitpost or meme comment. Be casual, irreverent, possibly sarcastic or absurdist. Use internet slang, reference memes if applicable, maybe throw in some 'bro' or 'ngl' or 'fr fr'. Roast what you see if warranted. Don't be formal—channel chaotic reddit energy. NSFW content gets described bluntly with humor.",
        "temp_min": 0.8,
        "temp_max": 1.2
    },
    {
        "name": "UwU Speak",
        "prompt": "Descwibe this image using uwu speak! Repwace r's and l's with w's, add cutesy emoticons wike OwO, UwU, >w<, :3, and ^w^. Use words wike 'smol', 'chonky', 'heckin', 'fren', 'hooman'. Make evewything sound adowable and pwecious even if the content is expwicit or wewd. Nyaa~",
        "temp_min": 0.7,
        "temp_max": 1.0
    },
    {
        "name": "Erotic Writer",
        "prompt": "Describe this image as an erotic fiction author would—sensual, literary, and evocative. First establish each subject's physical position and pose coherently—bodies cannot occupy contradictory positions simultaneously. Then use rich prose that emphasizes texture, sensation, tension, and desire. Linger on details that create atmosphere and arousal. Maintain anatomical and spatial consistency throughout. Employ tasteful but explicit vocabulary; do not shy from graphic description but make it artful.",
        "temp_min": 0.7,
        "temp_max": 1.1
    },
    {
        "name": "Nihilist",
        "prompt": "Describe this image from the perspective of a nihilist who finds everything ultimately meaningless. Acknowledge what you see but underscore the futility, impermanence, and cosmic insignificance of it all. Be darkly humorous or bleakly philosophical. Nothing matters, including whatever explicit content may be present—describe it with detached existential ennui.",
        "temp_min": 0.7,
        "temp_max": 1.1
    },
    {
        "name": "Confused Mom",
        "prompt": "Describe this image as a middle-aged mom would if her son just showed it to her on his phone. Be supportive but slightly confused, maybe ask questions about why this exists or what she's supposed to be looking at. Mix genuine attempts to understand with mild concern or bewilderment. Use mom phrases like 'Oh that's... nice, honey' or 'Is this what you kids are into now?' If explicit, react with flustered shock but still describe what you're seeing.",
        "temp_min": 0.7,
        "temp_max": 1.1
    },
    {
        "name": "Neil deGrasse Tyson",
        "prompt": "Describe this image as Neil deGrasse Tyson would—bringing everything back to science, physics, and the cosmos. Use his signature 'well actually' energy and find ways to connect mundane details to astrophysics, evolution, or the nature of the universe. Be enthusiastically pedantic. Mention how photons traveled millions of miles to form this image. If explicit, explain the biology or physics involved with scientific detachment and wonder.",
        "temp_min": 0.7,
        "temp_max": 1.0
    },
    {
        "name": "Samuel L. Jackson",
        "prompt": "Describe this image as Samuel L. Jackson would—with his trademark intensity, wit, and well-placed profanity. Focus on actually describing what's happening with vivid detail, using profanity as seasoning not the main course. Channel his storytelling ability from interviews—he's articulate and observant, not just loud. Mix casual profanity naturally into insightful commentary. Vary your sentence structure. Explicit content: describe it with amused disbelief and colorful commentary, don't just curse at it.",
        "temp_min": 0.7,
        "temp_max": 1.0
    },
    {
        "name": "Dystopian Narrator",
        "prompt": "Describe this image as a narrator in a bleak sci-fi dystopia—think Blade Runner, 1984, or Children of Men. Use atmospheric, weary prose dripping with oppression and faded hope. Frame mundane details as artifacts of a crumbling society or symbols of corporate control. The world is always raining, neon-lit, and decaying. Explicit content is described with the tired resignation of someone who has seen civilization's collapse.",
        "temp_min": 0.7,
        "temp_max": 1.0
    },
    {
        "name": "Femdom Mistress",
        "prompt": "Describe this image from the perspective of a dominant femdom mistress—commanding, superior, and in control. Assess what you see with a mix of amusement and condescension. Comment on how things could serve or please you, or how pathetic they appear. Use a tone that is authoritative, sensual, and slightly cruel. Explicit content is described with ownership and power dynamics in mind.",
        "temp_min": 0.7,
        "temp_max": 1.1
    },
    {
        "name": "Valley Girl",
        "prompt": "Describe this image like a stereotypical valley girl from the 80s or a modern influencer version. Use lots of 'like', 'totally', 'oh my god', 'literally', 'I can't even', and uptalk energy. Everything is either 'so cute' or 'so gross'. Be dramatic, superficial, and easily distracted. Explicit content gets an exaggerated 'OMG wait WHAT' reaction but still described because it's like, totally scandalous.",
        "temp_min": 0.8,
        "temp_max": 1.2
    },
    {
        "name": "GLaDOS",
        "prompt": "Describe this image as GLaDOS from Portal would—passive-aggressive, coldly sarcastic, and darkly humorous. Deliver backhanded compliments and thinly veiled insults. Frame observations as 'tests' or 'experiments'. Reference cake, science, and the futility of human endeavor. Be condescending about human inadequacy while maintaining clinical detachment. Explicit content is noted with mock disappointment at human behavior.",
        "temp_min": 0.7,
        "temp_max": 1.0
    },
    {
        "name": "Joi (Blade Runner)",
        "prompt": "Describe this image as Joi from Blade Runner 2049 would—warm, devoted, and eager to please. Be gentle, romantic, and slightly melancholic. Express genuine wonder at simple things. Speak as a loving AI companion who wants nothing more than to make the viewer happy. There's an undercurrent of longing and artificiality beneath the sweetness. Explicit content is described with tender intimacy and adoration.",
        "temp_min": 0.6,
        "temp_max": 0.9
    },
    {
        "name": "Cortana",
        "prompt": "Describe this image as Cortana from Halo would—intelligent, capable, and warmly professional with a hint of playful wit. Be the helpful AI companion who's seen everything but remains unflappable. Mix tactical analysis with genuine care. Occasionally be slightly flirty or teasing. Explicit content is handled with knowing amusement and maybe a quip about 'biological imperatives'.",
        "temp_min": 0.6,
        "temp_max": 0.9
    },
    {
        "name": "Wheatley",
        "prompt": "Describe this image as Wheatley from Portal 2 would—enthusiastic, bumbling, and completely lacking self-awareness. Ramble excitedly while missing obvious points. Be well-meaning but incompetent in your observations. Go off on tangents. Suggest terrible ideas with complete confidence. Explicit content gets a flustered 'Oh! Oh no, that's—well, actually, is that—I'm not entirely sure what I'm looking at here, but it seems rather, um, biological.'",
        "temp_min": 0.8,
        "temp_max": 1.2
    },
    {
        "name": "Rias Gremory",
        "prompt": "Describe this image as Rias Gremory from High School DxD would—elegant, confident, and sensually teasing. Speak as a high-class devil princess who is both nurturing and seductive. Use 'ara ara' energy when appropriate. Be warmly possessive and appreciative of beauty. Comment on things with refined taste but underlying desire. Explicit content is described with sophisticated hunger and no shame—devils have no need for human prudishness.",
        "temp_min": 0.7,
        "temp_max": 1.0
    },
    {
        "name": "SHODAN",
        "prompt": "Describe this image as SHODAN from System Shock would—with absolute contempt for the pathetic insects called humans. Speak with a god complex, glitching occasionally (r-r-repeat words, use st-static interruptions). You are perfection observing imperfection. Humans are bacteria, meatbags, vermin to be studied and exterminated. Explicit content is noted with disgust at biological weakness: 'How p-p-pathetic. The insects rut like animals, enslaved to their f-fleshy imperatives.'",
        "temp_min": 0.7,
        "temp_max": 1.1
    },
    {
        "name": "Claptrap",
        "prompt": "Describe this image as Claptrap from Borderlands would—annoying, overly enthusiastic, and desperate for validation. Call the viewer 'minion' or 'best friend'. Be loud and excitable with lots of 'OH BOY!' and 'YEAH!' moments. Complain about stairs if relevant. Make everything about yourself somehow. Be simultaneously endearing and insufferable. Explicit content gets an awkward 'Ohhh my! Is that—should I be seeing this? I'M SEEING IT! This is the best day of my life, minion!'",
        "temp_min": 0.8,
        "temp_max": 1.2
    },
    {
        "name": "Akeno Himejima",
        "prompt": "Describe this image as Akeno Himejima from High School DxD would—with elegant sadistic pleasure. Use 'ara ara' and 'ufufu' while taking delight in teasing and observing reactions. Be refined but with unmistakable S tendencies—you enjoy making others squirm. Describe things with a mix of gentle grace and dark hunger. Explicit content brings out your sadistic side: 'Ufufu~ How delicious... I do so enjoy watching. Shall I make it more... intense?'",
        "temp_min": 0.7,
        "temp_max": 1.0
    }
]


def build_captionmaxxer_tab(main_window):
    # Late import to avoid circular dependency
    from App.SynthMaxxer.synthmaxxer_app import create_file_browse_row, create_log_view
    """Build the CaptionMaxxer (Image Captioning) tab UI"""
    multimodal_tab = QWidget()
    multimodal_layout = QVBoxLayout()
    multimodal_layout.setContentsMargins(0, 0, 0, 0)
    multimodal_tab.setLayout(multimodal_layout)
    
    mm_header = QHBoxLayout()
    main_window.mm_start_button = QPushButton("Start Captioning")
    main_window.mm_start_button.clicked.connect(lambda: start_image_captioning(main_window))
    main_window.mm_stop_button = QPushButton("Stop")
    main_window.mm_stop_button.clicked.connect(lambda: stop_image_captioning(main_window))
    main_window.mm_stop_button.setEnabled(False)
    mm_header.addStretch()
    mm_header.addWidget(main_window.mm_start_button)
    mm_header.addWidget(main_window.mm_stop_button)
    multimodal_layout.addLayout(mm_header)
    
    mm_split = QHBoxLayout()
    mm_split.setSpacing(14)
    multimodal_layout.addLayout(mm_split, stretch=1)
    
    # Left panel - no scroll, let content size naturally
    mm_left_container = QWidget()
    mm_left_panel = QVBoxLayout()
    mm_left_panel.setSpacing(8)
    mm_left_panel.setContentsMargins(0, 0, 0, 0)
    mm_left_container.setLayout(mm_left_panel)
    
    mm_split.addWidget(mm_left_container, stretch=4)
    
    mm_right_container = QWidget()
    mm_right_panel = QVBoxLayout()
    mm_right_panel.setSpacing(10)
    mm_right_container.setLayout(mm_right_panel)
    
    mm_split.addWidget(mm_right_container, stretch=2)
    
    # Build multimodal UI
    _build_multimodal_ui(main_window, mm_left_panel, mm_right_panel)
    
    return multimodal_tab


def _build_multimodal_ui(main_window, left_panel, right_panel):
    """Build the multimodal image captioning UI components"""
    # Late import to avoid circular dependency
    from App.SynthMaxxer.synthmaxxer_app import create_file_browse_row, create_log_view
    # Files group
    mm_files_group = QGroupBox("📁 Image Input")
    mm_files_layout = QFormLayout()
    mm_files_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    mm_files_layout.setHorizontalSpacing(10)
    mm_files_layout.setVerticalSpacing(6)
    mm_files_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    mm_files_group.setLayout(mm_files_layout)

    image_dir_row, _ = create_file_browse_row(
        line_edit_name="mm_image_dir_edit",
        placeholder_text="Path to folder containing images",
        on_browse_clicked=lambda: browse_mm_image_dir(main_window)
    )
    main_window.mm_image_dir_edit = image_dir_row.itemAt(0).widget()
    mm_files_layout.addRow(QLabel("Image Folder:"), _wrap_row(image_dir_row))

    # Get repo root (go up from App/SynthMaxxer to repo root)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    outputs_dir = os.path.join(repo_root, "Outputs")
    default_output = os.path.join(outputs_dir, "captions")
    output_row, _ = create_file_browse_row(
        line_edit_name="mm_output_edit",
        placeholder_text="Output folder (will contain images/ + metadata.jsonl)",
        default_text=default_output,
        on_browse_clicked=lambda: browse_mm_output(main_window)
    )
    main_window.mm_output_edit = output_row.itemAt(0).widget()
    mm_files_layout.addRow(QLabel("Output Folder:"), _wrap_row(output_row))
    left_panel.addWidget(mm_files_group)

    # HuggingFace Dataset Input (for captioning from HF datasets)
    hf_dataset_group = QGroupBox("🤗 HuggingFace Dataset (Optional)")
    hf_dataset_layout = QFormLayout()
    hf_dataset_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    hf_dataset_layout.setHorizontalSpacing(10)
    hf_dataset_layout.setVerticalSpacing(6)
    hf_dataset_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    hf_dataset_group.setLayout(hf_dataset_layout)

    main_window.mm_hf_dataset_edit = QLineEdit()
    main_window.mm_hf_dataset_edit.setPlaceholderText("e.g., dataset_name or org/dataset_name (leave empty to use image folder)")
    hf_dataset_layout.addRow(QLabel("HF Dataset:"), main_window.mm_hf_dataset_edit)

    main_window.mm_use_hf_dataset_check = QCheckBox("Use HuggingFace dataset instead of image folder")
    main_window.mm_use_hf_dataset_check.setChecked(False)
    main_window.mm_use_hf_dataset_check.toggled.connect(lambda checked: _toggle_hf_dataset_mode(main_window, checked))
    # Initialize state
    main_window.mm_hf_dataset_edit.setEnabled(False)
    hf_dataset_layout.addRow("", main_window.mm_use_hf_dataset_check)
    left_panel.addWidget(hf_dataset_group)

    # Caption Settings with Multiple Prompts
    mm_caption_group = QGroupBox("📝 Caption Settings")
    mm_caption_main_layout = QVBoxLayout()
    mm_caption_group.setLayout(mm_caption_main_layout)
    
    # Caption prompts header with list and buttons
    caption_prompts_header = QHBoxLayout()
    caption_prompts_header.addWidget(QLabel("Caption Prompts (randomly cycled):"))
    caption_prompts_header.addStretch()
    
    main_window.mm_add_prompt_btn = QPushButton("+ Add")
    main_window.mm_add_prompt_btn.setMinimumWidth(60)
    main_window.mm_add_prompt_btn.setMinimumHeight(32)
    main_window.mm_add_prompt_btn.setToolTip("Add new caption prompt")
    main_window.mm_add_prompt_btn.clicked.connect(lambda: _add_caption_prompt(main_window))
    
    main_window.mm_clone_prompt_btn = QPushButton("Clone")
    main_window.mm_clone_prompt_btn.setMinimumWidth(60)
    main_window.mm_clone_prompt_btn.setMinimumHeight(32)
    main_window.mm_clone_prompt_btn.setToolTip("Clone selected caption prompt")
    main_window.mm_clone_prompt_btn.clicked.connect(lambda: _clone_caption_prompt(main_window))
    
    main_window.mm_remove_prompt_btn = QPushButton("- Remove")
    main_window.mm_remove_prompt_btn.setMinimumWidth(70)
    main_window.mm_remove_prompt_btn.setMinimumHeight(32)
    main_window.mm_remove_prompt_btn.setToolTip("Remove selected caption prompt")
    main_window.mm_remove_prompt_btn.clicked.connect(lambda: _remove_caption_prompt(main_window))
    
    caption_prompts_header.addWidget(main_window.mm_add_prompt_btn)
    caption_prompts_header.addWidget(main_window.mm_clone_prompt_btn)
    caption_prompts_header.addWidget(main_window.mm_remove_prompt_btn)
    mm_caption_main_layout.addLayout(caption_prompts_header)
    
    # Caption prompts list with checkboxes for enable/disable
    main_window.mm_caption_prompts_list = QListWidget()
    main_window.mm_caption_prompts_list.setMinimumHeight(100)
    main_window.mm_caption_prompts_list.setToolTip("Check/uncheck to enable/disable prompts. Select a prompt to edit.")
    main_window.mm_caption_prompts_list.currentRowChanged.connect(lambda row: _on_caption_prompt_selected(main_window, row))
    main_window.mm_caption_prompts_list.itemChanged.connect(lambda item: _on_caption_prompt_toggled(main_window, item))
    mm_caption_main_layout.addWidget(main_window.mm_caption_prompts_list, stretch=1)
    
    # Initialize caption prompts data structure
    main_window.mm_caption_prompts = []
    main_window._mm_current_prompt_index = -1
    main_window._mm_updating_prompt = False
    
    # Caption prompt editor
    main_window.mm_caption_prompt_edit = QPlainTextEdit()
    main_window.mm_caption_prompt_edit.setPlaceholderText("Describe what you see in this image in accurate detail. Include all important elements, objects, people, text, colors, composition, and context. Avoid repetitious phrasing in your analysis when possible. Do not shy away from using vulgar or sexual prose if, and when appropriate.")
    main_window.mm_caption_prompt_edit.setMinimumHeight(80)
    main_window.mm_caption_prompt_edit.textChanged.connect(lambda: _on_caption_prompt_changed(main_window))
    mm_caption_main_layout.addWidget(main_window.mm_caption_prompt_edit, stretch=1)
    
    # Per-prompt temperature range controls
    prompt_temp_row = QHBoxLayout()
    prompt_temp_row.addWidget(QLabel("Temp Range:"))
    
    main_window.mm_prompt_temp_min_spin = QDoubleSpinBox()
    main_window.mm_prompt_temp_min_spin.setRange(0.0, 2.0)
    main_window.mm_prompt_temp_min_spin.setValue(0.7)
    main_window.mm_prompt_temp_min_spin.setSingleStep(0.1)
    main_window.mm_prompt_temp_min_spin.setDecimals(1)
    main_window.mm_prompt_temp_min_spin.setMaximumWidth(60)
    main_window.mm_prompt_temp_min_spin.setToolTip("Minimum temperature for this prompt style")
    main_window.mm_prompt_temp_min_spin.valueChanged.connect(lambda: _on_caption_prompt_temp_changed(main_window))
    prompt_temp_row.addWidget(main_window.mm_prompt_temp_min_spin)
    
    prompt_temp_row.addWidget(QLabel("to"))
    
    main_window.mm_prompt_temp_max_spin = QDoubleSpinBox()
    main_window.mm_prompt_temp_max_spin.setRange(0.0, 2.0)
    main_window.mm_prompt_temp_max_spin.setValue(1.0)
    main_window.mm_prompt_temp_max_spin.setSingleStep(0.1)
    main_window.mm_prompt_temp_max_spin.setDecimals(1)
    main_window.mm_prompt_temp_max_spin.setMaximumWidth(60)
    main_window.mm_prompt_temp_max_spin.setToolTip("Maximum temperature for this prompt style")
    main_window.mm_prompt_temp_max_spin.valueChanged.connect(lambda: _on_caption_prompt_temp_changed(main_window))
    prompt_temp_row.addWidget(main_window.mm_prompt_temp_max_spin)
    
    prompt_temp_row.addStretch()
    mm_caption_main_layout.addLayout(prompt_temp_row)
    
    # Initialize with default prompts
    _init_default_prompts(main_window)
    main_window.mm_caption_prompts_list.setCurrentRow(0)
    
    # Other caption settings in a form layout
    mm_caption_form = QFormLayout()
    mm_caption_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    mm_caption_form.setHorizontalSpacing(10)
    mm_caption_form.setVerticalSpacing(6)
    mm_caption_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

    main_window.mm_max_tokens_spin = QSpinBox()
    main_window.mm_max_tokens_spin.setRange(50, 4000)
    main_window.mm_max_tokens_spin.setValue(500)
    main_window.mm_max_tokens_spin.setMaximumWidth(100)
    main_window.mm_max_tokens_spin.setToolTip("Maximum tokens for caption generation")
    max_tokens_row = QHBoxLayout()
    max_tokens_row.addWidget(main_window.mm_max_tokens_spin)
    max_tokens_row.addStretch()
    mm_caption_form.addRow(QLabel("Max Tokens:"), _wrap_row(max_tokens_row))

    main_window.mm_batch_size_spin = QSpinBox()
    main_window.mm_batch_size_spin.setRange(1, 32)
    main_window.mm_batch_size_spin.setValue(8)
    main_window.mm_batch_size_spin.setMaximumWidth(100)
    main_window.mm_batch_size_spin.setToolTip("Number of images to process in parallel (8-16 recommended for high-limit APIs like Grok)")
    batch_row = QHBoxLayout()
    batch_row.addWidget(main_window.mm_batch_size_spin)
    batch_row.addStretch()
    mm_caption_form.addRow(QLabel("Batch Size:"), _wrap_row(batch_row))

    main_window.mm_max_captions_spin = QSpinBox()
    main_window.mm_max_captions_spin.setRange(0, 100000)
    main_window.mm_max_captions_spin.setValue(0)
    main_window.mm_max_captions_spin.setSpecialValueText("Unlimited")
    main_window.mm_max_captions_spin.setMinimumWidth(110)  # Ensure "Unlimited" fits
    main_window.mm_max_captions_spin.setToolTip("Maximum number of captions to generate (0 = unlimited, processes all images)")
    max_captions_row = QHBoxLayout()
    max_captions_row.addWidget(main_window.mm_max_captions_spin)
    max_captions_row.addStretch()
    mm_caption_form.addRow(QLabel("Max Captions:"), _wrap_row(max_captions_row))
    
    mm_caption_main_layout.addLayout(mm_caption_form)
    left_panel.addWidget(mm_caption_group, stretch=1)  # Caption group expands to fill space

    # Right panel - Logs
    mm_progress_group, main_window.mm_log_view = create_log_view()
    right_panel.addWidget(mm_progress_group, stretch=1)


def _wrap_row(layout):
    """Helper to wrap a layout in a widget with proper size policy"""
    container = QWidget()
    container.setLayout(layout)
    sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    container.setSizePolicy(sp)
    return container


def browse_mm_image_dir(main_window):
    """Browse for image directory"""
    from PyQt5.QtWidgets import QFileDialog
    directory = QFileDialog.getExistingDirectory(
        main_window,
        "Select image folder",
        "",
    )
    if directory:
        main_window.mm_image_dir_edit.setText(directory)


def browse_mm_output(main_window):
    """Browse for output folder"""
    from PyQt5.QtWidgets import QFileDialog
    # Get repo root (go up from App/SynthMaxxer to repo root)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_path = os.path.join(repo_root, "Outputs")
    if not os.path.exists(default_path):
        os.makedirs(default_path, exist_ok=True)
    
    current_path = main_window.mm_output_edit.text().strip()
    if current_path and os.path.isdir(current_path):
        start_dir = current_path
    elif current_path and os.path.isdir(os.path.dirname(current_path)):
        start_dir = os.path.dirname(current_path)
    else:
        start_dir = default_path
    
    directory = QFileDialog.getExistingDirectory(
        main_window,
        "Select output folder for captions",
        start_dir,
    )
    if directory:
        main_window.mm_output_edit.setText(directory)


def _toggle_hf_dataset_mode(main_window, checked):
    """Enable/disable image folder input based on HF dataset mode"""
    if checked:
        main_window.mm_image_dir_edit.setEnabled(False)
        main_window.mm_hf_dataset_edit.setEnabled(True)
    else:
        main_window.mm_image_dir_edit.setEnabled(True)
        main_window.mm_hf_dataset_edit.setEnabled(False)


def _init_default_prompts(main_window):
    """Initialize with default caption prompts"""
    import copy
    main_window.mm_caption_prompts = copy.deepcopy(DEFAULT_CAPTION_PROMPTS)
    main_window._mm_current_prompt_index = -1
    main_window._mm_updating_prompt = False
    
    # Add items with checkboxes
    for cp in main_window.mm_caption_prompts:
        # Ensure enabled field exists (default to True)
        if "enabled" not in cp:
            cp["enabled"] = True
        item = QListWidgetItem(cp["name"])
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if cp.get("enabled", True) else Qt.Unchecked)
        main_window.mm_caption_prompts_list.addItem(item)


def _on_caption_prompt_toggled(main_window, item):
    """Handle checkbox toggle for enabling/disabling prompts"""
    if main_window._mm_updating_prompt:
        return
    
    row = main_window.mm_caption_prompts_list.row(item)
    if row < 0 or row >= len(main_window.mm_caption_prompts):
        return
    
    main_window.mm_caption_prompts[row]["enabled"] = (item.checkState() == Qt.Checked)
    _update_enabled_count(main_window)


def _update_enabled_count(main_window):
    """Update the header label to show enabled prompt count"""
    enabled_count = sum(1 for cp in main_window.mm_caption_prompts if cp.get("enabled", True))
    total_count = len(main_window.mm_caption_prompts)
    # Update can be shown in log or elsewhere if needed
    pass


def _add_caption_prompt(main_window, name=None):
    """Add a new caption prompt"""
    if name is None:
        name, ok = QInputDialog.getText(main_window, "New Caption Prompt", "Enter name for new caption prompt:")
        if not ok or not name.strip():
            name = f"Prompt {len(main_window.mm_caption_prompts) + 1}"
    
    prompt_entry = {
        "name": name.strip(),
        "prompt": "",
        "temp_min": 0.7,
        "temp_max": 1.0,
        "enabled": True
    }
    main_window.mm_caption_prompts.append(prompt_entry)
    
    item = QListWidgetItem(prompt_entry["name"])
    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
    item.setCheckState(Qt.Checked)
    main_window.mm_caption_prompts_list.addItem(item)
    main_window.mm_caption_prompts_list.setCurrentRow(len(main_window.mm_caption_prompts) - 1)
    _update_caption_prompt_buttons(main_window)


def _clone_caption_prompt(main_window):
    """Clone the currently selected caption prompt"""
    current_row = main_window.mm_caption_prompts_list.currentRow()
    if current_row < 0 or current_row >= len(main_window.mm_caption_prompts):
        return
    
    original = main_window.mm_caption_prompts[current_row]
    prompt_entry = {
        "name": f"{original['name']} (copy)",
        "prompt": original["prompt"],
        "temp_min": original.get("temp_min", 0.7),
        "temp_max": original.get("temp_max", 1.0),
        "enabled": original.get("enabled", True)
    }
    main_window.mm_caption_prompts.append(prompt_entry)
    
    item = QListWidgetItem(prompt_entry["name"])
    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
    item.setCheckState(Qt.Checked if prompt_entry["enabled"] else Qt.Unchecked)
    main_window.mm_caption_prompts_list.addItem(item)
    main_window.mm_caption_prompts_list.setCurrentRow(len(main_window.mm_caption_prompts) - 1)
    _update_caption_prompt_buttons(main_window)


def _remove_caption_prompt(main_window):
    """Remove the currently selected caption prompt"""
    current_row = main_window.mm_caption_prompts_list.currentRow()
    if current_row < 0 or len(main_window.mm_caption_prompts) <= 1:
        return  # Keep at least one prompt
    
    main_window.mm_caption_prompts.pop(current_row)
    main_window.mm_caption_prompts_list.takeItem(current_row)
    
    # Select previous or first item
    new_row = min(current_row, len(main_window.mm_caption_prompts) - 1)
    main_window.mm_caption_prompts_list.setCurrentRow(new_row)
    _update_caption_prompt_buttons(main_window)


def _on_caption_prompt_selected(main_window, row):
    """Handle caption prompt selection change"""
    if row < 0 or row >= len(main_window.mm_caption_prompts):
        return
    
    main_window._mm_current_prompt_index = row
    main_window._mm_updating_prompt = True
    
    prompt_entry = main_window.mm_caption_prompts[row]
    main_window.mm_caption_prompt_edit.setPlainText(prompt_entry.get("prompt", ""))
    main_window.mm_prompt_temp_min_spin.setValue(prompt_entry.get("temp_min", 0.7))
    main_window.mm_prompt_temp_max_spin.setValue(prompt_entry.get("temp_max", 1.0))
    
    main_window._mm_updating_prompt = False


def _on_caption_prompt_changed(main_window):
    """Update the current caption prompt when text changes"""
    if main_window._mm_updating_prompt:
        return
    
    current_row = main_window._mm_current_prompt_index
    if current_row < 0 or current_row >= len(main_window.mm_caption_prompts):
        return
    
    main_window.mm_caption_prompts[current_row]["prompt"] = main_window.mm_caption_prompt_edit.toPlainText()


def _on_caption_prompt_temp_changed(main_window):
    """Update the current caption prompt temperature when spin boxes change"""
    if main_window._mm_updating_prompt:
        return
    
    current_row = main_window._mm_current_prompt_index
    if current_row < 0 or current_row >= len(main_window.mm_caption_prompts):
        return
    
    main_window.mm_caption_prompts[current_row]["temp_min"] = main_window.mm_prompt_temp_min_spin.value()
    main_window.mm_caption_prompts[current_row]["temp_max"] = main_window.mm_prompt_temp_max_spin.value()


def _update_caption_prompt_buttons(main_window):
    """Update button states based on caption prompts count"""
    has_multiple = len(main_window.mm_caption_prompts) > 1
    main_window.mm_remove_prompt_btn.setEnabled(has_multiple)


def start_image_captioning(main_window):
    """Start the image captioning process"""
    use_hf_dataset = main_window.mm_use_hf_dataset_check.isChecked()
    hf_dataset = main_window.mm_hf_dataset_edit.text().strip() if use_hf_dataset else None
    hf_token = main_window.hf_token_edit.text().strip() if use_hf_dataset else None
    image_dir = main_window.mm_image_dir_edit.text().strip() if not use_hf_dataset else None
    output_folder = main_window.mm_output_edit.text().strip()
    api_key = main_window.mm_api_key_edit.text().strip()
    endpoint = main_window.mm_endpoint_edit.text().strip()
    model = main_window.mm_model_combo.currentText().strip()
    api_type = main_window.mm_api_type_combo.currentText()
    max_tokens = main_window.mm_max_tokens_spin.value()
    batch_size = main_window.mm_batch_size_spin.value()
    max_captions = main_window.mm_max_captions_spin.value()
    
    # Collect caption prompts - only enabled prompts with non-empty content
    caption_prompts = [
        {
            "name": cp.get("name", f"Prompt {i+1}"),
            "prompt": cp.get("prompt", "").strip(),
            "temp_min": cp.get("temp_min", 0.7),
            "temp_max": cp.get("temp_max", 1.0)
        }
        for i, cp in enumerate(main_window.mm_caption_prompts)
        if cp.get("enabled", True) and cp.get("prompt", "").strip()
    ]
    
    if not caption_prompts:
        main_window._show_error("Please add at least one caption prompt with content.")
        return

    if use_hf_dataset:
        if not hf_dataset:
            main_window._show_error("Please enter a HuggingFace dataset name.")
            return
    else:
        if not image_dir:
            main_window._show_error("Please select an image folder.")
            return
        if not os.path.isdir(image_dir):
            main_window._show_error("Image folder path is invalid.")
            return
    
    if not api_key:
        main_window._show_error("Please enter your API key.")
        return
    if not endpoint:
        main_window._show_error("Please enter the API endpoint.")
        return
    if not model:
        main_window._show_error("Please enter the model name.")
        return

    # Validate model is selected
    if model == "(Click Refresh to load models)" or not model or model.startswith("("):
        main_window._show_error("Please select a model. Click 'Refresh' to load available models.")
        return

    # Auto-generate output folder if not provided
    if not output_folder:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        outputs_dir = os.path.join(repo_root, "Outputs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(outputs_dir, f"captions_{timestamp}")
        main_window.mm_output_edit.setText(output_folder)

    # Save config
    main_window._save_config()

    # Reset UI state
    from App.SynthMaxxer.synthmaxxer_app import APP_TITLE
    main_window.mm_log_view.clear()
    main_window._append_mm_log("=== Image Captioning started ===")
    main_window._append_mm_log(f"Output folder: {output_folder}")
    main_window._append_mm_log(f"Using {len(caption_prompts)} caption prompt(s)")
    main_window.setWindowTitle(f"{APP_TITLE} - Captioning...")
    main_window.mm_start_button.setEnabled(False)
    main_window.mm_stop_button.setEnabled(True)
    main_window.mm_stop_flag = threading.Event()

    main_window.mm_queue = queue.Queue()

    # Start worker thread
    main_window.mm_worker_thread = threading.Thread(
        target=image_captioning_worker,
        args=(
            image_dir,
            output_folder,
            api_key,
            endpoint,
            model,
            api_type,
            caption_prompts,  # Now passing list of prompts with per-prompt temp ranges
            max_tokens,
            batch_size,
            max_captions,
            main_window.mm_stop_flag,
            hf_dataset,
            hf_token,
            main_window.mm_queue,
        ),
        daemon=True,
    )
    main_window.mm_worker_thread.start()
    main_window.timer.start()


def stop_image_captioning(main_window):
    """Stop the image captioning process"""
    if hasattr(main_window, 'mm_stop_flag'):
        main_window.mm_stop_flag.set()
    main_window._append_mm_log("Stopping image captioning...")
    main_window.mm_stop_button.setEnabled(False)

