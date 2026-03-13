import gradio as gr
import requests
import json
import torch
import textwrap
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 1. SETUP STABLE DIFFUSION FOR MAC CPU
# ==========================================
print("Loading Stable Diffusion 1.5 to CPU...")
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float32,
    safety_checker=None
).to("cpu")

STYLE_PROMPT = "Vintage Amar Chitra Katha comic book style, 2d flat colors, cel shaded, strong bold black ink outlines, clear human characters, Indian mythology, simple minimalist background, graphic novel illustration"

NEGATIVE_PROMPT = "photorealistic, 3d, cinematic, realistic lighting, complex textures, gradient, abstract, text, watermark, speech bubble"

# ==========================================
# 2. OLLAMA LLM FUNCTIONS
# ==========================================
def ask_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {"model": "llama3.1", "prompt": prompt, "stream": False}
    try:
        response = requests.post(url, json=payload)
        return response.json()['response']
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}. Make sure Ollama is running."

def generate_story(dilemma):
    prompt = f"""You are a sage retelling the Mahabharata for modern youth.
    Modern dilemma: '{dilemma}'
    Rewrite it as a 3-paragraph mythological parable. Use characters like Arjuna or Krishna.
    End with a timeless lesson."""
    return ask_ollama(prompt)

def generate_panel_descriptions(story):
    prompt = f"""Break this story into EXACTLY 4 comic panels.
    Return ONLY a raw JSON list of 4 objects with keys:
    "panel": (1-4),
    "description": (An extremely literal visual description for an AI image generator. Describe the characters physically and what they are doing. E.g., instead of 'Arjuna feels sad', write 'A young Indian warrior in traditional armor holding a bow, looking down in despair'.),
    "dialogue": (A short, punchy 5-10 word Marvel-style quote or thought to go inside a speech bubble)
    Story: {story}"""
    
    response = ask_ollama(prompt)
    try:
        json_str = response[response.find("["):response.rfind("]")+1]
        panels = json.loads(json_str)[:4]
    except:
        panels = [{"panel": i, "description": f"A traditional Indian warrior talking to a wise sage in a folk art style", "dialogue": "What should I do?"} for i in range(1,5)]
    return panels

# ==========================================
# 3. DRAW MARVEL-STYLE SPEECH BUBBLES
# ==========================================
def draw_speech_bubble(image, text):
    """Draws a comic-book style speech bubble on the image"""
    draw = ImageDraw.Draw(image)
    
    # Try to load a standard Mac font, fallback to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Wrap the text so it fits in the bubble
    wrapped_text = textwrap.fill(text, width=25)
    
    # Calculate text bounding box
    bbox = draw.textbbox((0, 0), wrapped_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Bubble dimensions and padding
    padding = 15
    bubble_x1 = 20
    bubble_y1 = 20
    bubble_x2 = bubble_x1 + text_width + (padding * 2)
    bubble_y2 = bubble_y1 + text_height + (padding * 2)
    
    # Draw the main bubble (white with black outline)
    draw.rounded_rectangle(
        [bubble_x1, bubble_y1, bubble_x2, bubble_y2], 
        radius=15, fill="white", outline="black", width=3
    )
    
    # Draw the little "tail" of the speech bubble to make it look authentic
    tail_coords = [
        (bubble_x1 + 20, bubble_y2), 
        (bubble_x1 + 40, bubble_y2), 
        (bubble_x1 + 10, bubble_y2 + 20)
    ]
    draw.polygon(tail_coords, fill="white", outline="black")
    # Hide the line dividing the tail and the bubble
    draw.line([(bubble_x1 + 21, bubble_y2), (bubble_x1 + 39, bubble_y2)], fill="white", width=3)
    
    # Draw the text inside
    draw.text((bubble_x1 + padding, bubble_y1 + padding), wrapped_text, fill="black", font=font)
    return image

# ==========================================
# 4. MAIN GENERATOR FUNCTION FOR UI
# ==========================================
def generate_comic_ui(dilemma):
    yield "Generating story via Ollama...", None, None
    
    story = generate_story(dilemma)
    if "Error" in story:
        yield story, None, None
        return
        
    yield "Story generated! Parsing panels...", story, None
    panels = generate_panel_descriptions(story)
    
    images = []
    for i, p in enumerate(panels):
        yield f"🎨 Drawing panel {i+1}/4 (This takes time on CPU)...", story, None
        
        full_prompt = f"{STYLE_PROMPT}, {p['description']}"
        base_image = pipe(
            full_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=12, # REDUCED from 15 or 20! This will speed up CPU generation significantly.
            guidance_scale=7.0,     # Slightly lower guidance also helps with flatter images
            height=512,
            width=512
        ).images[0]
        
        # Add the speech bubble to the generated image
        final_image = draw_speech_bubble(base_image, p.get("dialogue", "..."))
        images.append(final_image)

    yield "🖼️ Stitching final comic layout...", story, None
    
    # Stitch 2x2 grid
    grid = Image.new('RGB', (1024, 1024), color=(245, 222, 179))
    for idx, img in enumerate(images):
        grid.paste(img, ((idx % 2) * 512, (idx // 2) * 512))
        
    grid.save("intel_mac_comic_final.png")
    
    yield "✅ Complete!", story, "intel_mac_comic_final.png"

# ==========================================
# 5. GRADIO UI LAYOUT
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🪷 Cultural Mythology Reteller & Comic Generator")
    gr.Markdown("Enter a modern dilemma, and watch it transform into a traditional Indian comic strip.")
    
    with gr.Row():
        with gr.Column(scale=1):
            dilemma_input = gr.Textbox(
                label="Your Modern Dilemma", 
                placeholder="e.g., I'm addicted to doomscrolling on social media...",
                lines=3
            )
            generate_btn = gr.Button("Generate Comic Strip", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False)
            story_output = gr.Textbox(label="The Mythological Parable", lines=10, interactive=False)
            
        with gr.Column(scale=2):
            comic_output = gr.Image(label="Generated Comic Strip", type="filepath")

    # Connect the button to the generator function using yield for live updates
    generate_btn.click(
        fn=generate_comic_ui,
        inputs=[dilemma_input],
        outputs=[status_text, story_output, comic_output]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)