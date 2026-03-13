# 🪷 Cultural Mythology Reteller & Comic Strip Generator

An AI-powered, full-stack application that transforms modern-day dilemmas into ancient mythological parables and generates a complete, Marvel-style comic strip layout. 

This project bridges the gap between ancient wisdom (like the Mahabharata and Ramayana) and contemporary struggles, utilizing local Large Language Models (LLMs) and Stable Diffusion.

![App Screenshot](intel_mac_comic_final(1).png) 

## 🌍 Why This Project Matters
In today's fast-paced, hyper-digital world, younger generations are increasingly losing connection to foundational cultural epics. Modern problems—like career anxiety, social media addiction, or imposter syndrome—often feel completely disconnected from ancient texts. 

This project solves that by:
1. **Contextualizing Wisdom:** Taking a user's modern dilemma and using **Llama 3.1** to rewrite it as a mythological parable featuring figures like Arjuna or Krishna.
2. **Visual Storytelling:** Automatically generating a 4-panel comic strip using **Stable Diffusion**, complete with dynamic speech bubbles and classic stylized comic art.
3. **Cultural Preservation:** Making ancient philosophy accessible, personalized, and visually engaging for schools, NGOs, and the general public.

## ⚙️ Architecture & Tech Stack
This application is designed to run entirely locally, specifically optimized for CPU-bound environments (like Intel Macs).
* **LLM:** [Ollama](https://ollama.com/) running `llama3.1` for story generation, panel parsing, and dialogue extraction.
* **Image Generation:** `runwayml/stable-diffusion-v1-5` running locally via Hugging Face `diffusers`.
* **Image Processing:** Python `Pillow` (PIL) for dynamic comic grid stitching and Marvel-style speech bubble generation.
* **User Interface:** `Gradio` for a seamless, interactive web application.

## 🚀 Setup & Installation Guide (Intel Mac / CPU Optimized)

Because modern AI tools heavily favor Nvidia GPUs or Apple Silicon (M-series), setting this up on an older x86_64 architecture requires specific versioning. 

### 1. Prerequisites
* **Python 3.11** (PyTorch 2.2.2 does not support Python 3.12+ for Intel Macs).
* **Ollama** installed on your system.

### 2. Start Ollama
Since Intel Macs cannot use Apple's MLX framework, force Ollama to use the standard CPU runner:
```bash
export OLLAMA_LLM_LIBRARY=cpu_avx2
ollama serve
