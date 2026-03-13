import requests
import json
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF
import time

# ==========================================
# 1. SETUP STABLE DIFFUSION FOR MAC CPU
# ==========================================
print("Loading Stable Diffusion 1.5 (This will take a moment...)")
# Using SD 1.5 instead of SDXL because SDXL will freeze an Intel CPU
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float32 # Intel Macs must use float32
)
pipe = pipe.to("cpu") # Force CPU usage
# Optional: pipe.enable_attention_slicing() # Uncomment if you run out of RAM

# ==========================================
# 2. PROMPTS
# ==========================================
STYLE_PROMPT = "Indian traditional art style, Madhubani painting, Warli fusion, tribal art, 2d flat, intricate patterns"
NEGATIVE_PROMPT = "photorealistic, 3d, modern, bad anatomy, text, watermark"

# ==========================================
# 3. OLLAMA LLM FUNCTIONS
# ==========================================
def ask_ollama(prompt):
    """Sends a prompt to the local Ollama application running Llama 3.1"""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.1",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    return response.json()['response']

def generate_story(dilemma):
    prompt = f"""You are a sage retelling the Mahabharata for modern youth.
    Take this modern dilemma: '{dilemma}'
    Rewrite it as a 3-paragraph mythological parable. Use characters like Arjuna, Krishna, or Karna.
    End with a timeless lesson."""
    return ask_ollama(prompt)

def generate_panel_descriptions(story):
    prompt = f"""Break this story into EXACTLY 4 comic panels.
    Return ONLY a raw JSON list of 4 objects with keys "panel" (1-4) and "description" (one sentence visual description).
    Story: {story}"""
    
    response = ask_ollama(prompt)
    
    # Try to extract just the JSON part in case the LLM adds conversational text
    try:
        json_str = response[response.find("["):response.rfind("]")+1]
        panels = json.loads(json_str)[:4]
    except:
        print("Warning: Failed to parse JSON, using fallback panels.")
        panels = [{"panel": i, "description": f"Mahabharata scene {i} showing traditional Indian art"} for i in range(1,5)]
    return panels

# ==========================================
# 4. MAIN GENERATOR
# ==========================================
def generate_comic(dilemma):
    print(f"\n🚀 Starting generation for: {dilemma}")

    # Step 1: Story
    print("✍️ Generating story via Ollama...")
    story = generate_story(dilemma)
    print("✅ Story generated!")

    # Step 2: Panels
    panels = generate_panel_descriptions(story)
    print(f"✅ {len(panels)} panels parsed!")

    # Step 3: Images (This will take time on an Intel CPU)
    images = []
    for i, p in enumerate(panels):
        full_prompt = f"{STYLE_PROMPT}, {p['description']}"
        print(f"🎨 Drawing panel {i+1}/4 (Grab a coffee, CPU rendering is slow)...")
        
        start_time = time.time()
        # Reduced inference steps to 15 to save CPU time
        image = pipe(
            full_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=15, 
            guidance_scale=7.5,
            height=512, # 512x512 is the max you should do on CPU
            width=512
        ).images[0]
        
        images.append(image)
        print(f"   ⏱️ Panel {i+1} took {round(time.time() - start_time, 1)} seconds.")

    # Step 4: Build Grid
    print("🖼️ Stitching comic grid...")
    grid = Image.new('RGB', (1024, 1024), color=(245, 222, 179))
    for idx, img in enumerate(images):
        x = (idx % 2) * 512
        y = (idx // 2) * 512
        grid.paste(img, (x, y))

    grid.save("intel_mac_comic.png")
    print("✅ Saved intel_mac_comic.png")

    # Step 5: PDF
    print("📄 Generating PDF...")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Mahabharata Retold", ln=1, align='C')
    
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, story)
    
    pdf.add_page()
    pdf.image("intel_mac_comic.png", x=10, y=10, w=190)
    pdf.output("story_comic.pdf")

    print("🎉 All done! Check your folder for the PDF and PNG.")

# Run the test
if __name__ == "__main__":
    generate_comic("AI is taking my job")