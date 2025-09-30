from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
import matplotlib.pyplot as plt
import json
from pydantic import BaseModel
from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg')

class AnalysisRequest(BaseModel):
    text: str
    model: str = "Mock Backend (Fast)"
    contradiction_threshold: float = 0.7
    hallucination_sensitivity: float = 0.8
    animation_enabled: bool = True
    physics_params: dict = {
        "c_semantic": 0.1,
        "c_attention": 0.01,
        "gamma_drag": 0.05
    }

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_attention_map(text: str) -> str:
    """Generate a more detailed attention heatmap."""
    # Create word-level attention scores
    words = text.split()
    n_words = len(words)
    attention_matrix = np.random.rand(n_words, n_words)
    
    # Create attention heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(attention_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Score')
    
    # Add word labels
    plt.xticks(range(n_words), words, rotation=45, ha='right')
    plt.yticks(range(n_words), words)
    
    plt.title('Word-Level Attention Map')
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_attention_gif(text: str) -> str:
    """Generate a more dynamic attention animation."""
    words = text.split()
    n_words = len(words)
    frames = []
    width, height = 500, 500
    n_frames = 10

    # Generate animation frames
    for frame in range(n_frames):
        img = Image.new('RGB', (width, height), color='black')
        draw = ImageDraw.Draw(img)
        
        # Create dynamic word positioning
        for i, word in enumerate(words):
            angle = (2 * np.pi * i / n_words) + (frame * 0.2)
            radius = 150 + 20 * np.sin(frame * 0.5)
            x = width/2 + radius * np.cos(angle)
            y = height/2 + radius * np.sin(angle)
            
            # Draw word
            draw.text((x, y), word, fill='white')
            
            # Draw connections
            if i > 0:
                prev_angle = (2 * np.pi * (i-1) / n_words) + (frame * 0.2)
                prev_x = width/2 + radius * np.cos(prev_angle)
                prev_y = height/2 + radius * np.sin(prev_angle)
                draw.line([(prev_x, prev_y), (x, y)], fill='blue', width=2)
        
        frames.append(img)
    
    # Save as animated GIF
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=100,
        loop=0
    )
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def analyze_text_integrity(text: str, contradiction_threshold: float, hallucination_sensitivity: float) -> Dict[str, List[Dict[str, Any]]]:
    """Perform more detailed text analysis."""
    words = text.split()
    
    # Analyze for hallucinations
    hallucinations = []
    potential_hallucinations = [
        "moon is made of cheese",
        "dragons exist",
        "time travel is common",
    ]
    for phrase in potential_hallucinations:
        if phrase.lower() in text.lower():
            hallucinations.append({
                "text": phrase,
                "confidence": round(np.random.uniform(0.7, 0.95), 2),
                "explanation": f"This statement lacks factual basis"
            })
    
    # Analyze for contradictions
    contradictions = []
    contradiction_pairs = [
        (["Paris", "Rome"], "location contradiction"),
        (["always", "never"], "temporal contradiction"),
        (["hot", "cold"], "property contradiction")
    ]
    for (words_pair, explanation) in contradiction_pairs:
        if all(word.lower() in text.lower() for word in words_pair):
            contradictions.append({
                "tokens": words_pair,
                "score": round(np.random.uniform(0.8, 0.95), 2),
                "explanation": explanation
            })
    
    # Analyze logical consistency
    logical_consistency = []
    if "miles" in text and "hours" in text:
        logical_consistency.append({
            "text": text,
            "is_consistent": True,
            "explanation": "Speed calculation is mathematically correct"
        })
    
    return {
        "hallucinations": hallucinations,
        "contradictions": contradictions,
        "logical_consistency": logical_consistency
    }

@app.get("/")
async def read_root():
    return {"message": "XAI-Phy API is running", "status": "healthy"}

@app.post("/analyze")
async def analyze_text(request: AnalysisRequest):
    try:
        # Generate visualizations
        attention_map = generate_attention_map(request.text)
        attention_gif = generate_attention_gif(request.text) if request.animation_enabled else None
        
        # Analyze text integrity
        integrity_report = analyze_text_integrity(
            request.text,
            request.contradiction_threshold,
            request.hallucination_sensitivity
        )
        
        return {
            "attention_map": attention_map,
            "attention_gif": attention_gif,
            "integrity_report": integrity_report,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))