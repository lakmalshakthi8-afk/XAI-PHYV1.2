from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib
import torch
from typing import List, Dict, Union, Optional, Any, Protocol
from numpy.typing import NDArray
from model_manager import ModelManager, MockBackend, GPT2Backend, BERTBackend, SentenceTransformerBackend

# Configure matplotlib for non-interactive backend
matplotlib.use('Agg')

# Type definitions
class ModelResult(Protocol):
    tokens: List[str]
    attention: NDArray[np.float64]
    embeddings: Optional[NDArray[np.float64]] = None

class IntegrityReport(Protocol):
    hallucinations: List[Dict[str, Any]]
    contradictions: List[Dict[str, Any]]
    logical_consistency: List[Dict[str, Any]]

class ModelBackend(Protocol):
    async def process_text(self, text: str) -> Dict[str, Any]:
        ...
    
    async def analyze_integrity(self, text: str, tokens: List[str], attention: NDArray[np.float64]) -> Dict[str, Any]:
        ...

app = FastAPI()

# Enable CORS with specific origins
origins = [
    "http://localhost:3000",
    "http://localhost:5000",
    "http://localhost:7071",
    "https://icy-island-0fa2fc70f.1.azurestaticapps.net",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = ModelManager()
current_model: str = "mock"  # Default to mock model
current_backend: Optional[ModelBackend] = None

async def init_model():
    global current_backend
    new_backend = model_manager.load_model(current_model)
    if new_backend is None or not isinstance(new_backend, ModelBackend):
        raise HTTPException(status_code=500, detail=f"Failed to load model {current_model}")
    current_backend = new_backend

@app.on_event("startup")
async def startup_event():
    await init_model()

def generate_attention_visualization(
    tokens: List[str],
    attention_matrix: NDArray[np.float64],
    title: str = "Attention Map"
) -> str:
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Attention Score')
    plt.title(title)
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    plt.tight_layout()
    
    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

def generate_attention_animation(
    tokens: List[str],
    attention_matrix: NDArray[np.float64],
    steps: int = 10
) -> str:
    frames = []
    # Initialize random positions for tokens
    positions = np.random.rand(len(tokens), 2) * 2 - 1
    velocities = np.zeros((len(tokens), 2))
    
    # Physics parameters
    dt = 0.1
    spring_k = 0.1
    drag = 0.1
    
    for step in range(steps):
        # Update positions based on attention
        forces = np.zeros_like(positions)
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                if i != j:
                    # Calculate spring force based on attention
                    direction = positions[j] - positions[i]
                    distance = np.linalg.norm(direction) + 1e-6
                    unit_direction = direction / distance
                    force = spring_k * attention_matrix[i, j] * unit_direction
                    forces[i] += force
        
        # Update velocities and positions
        velocities = velocities + forces * dt - drag * velocities
        positions = positions + velocities * dt
        
        # Create frame
        plt.figure(figsize=(10, 8))
        plt.scatter(positions[:, 0], positions[:, 1], s=1000)
        
        # Draw attention connections
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                if attention_matrix[i, j] > 0.1:
                    alpha = min(float(attention_matrix[i, j]), 0.8)
                    plt.plot([positions[i, 0], positions[j, 0]], 
                           [positions[i, 1], positions[j, 1]], 
                           'gray', alpha=alpha)
        
        # Add token labels
        for i, token in enumerate(tokens):
            plt.annotate(token, (positions[i, 0], positions[i, 1]), 
                        xytext=(0, 0), textcoords='offset points',
                        ha='center', va='center')
        
        plt.title(f'Token Relationships - Step {step + 1}')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        
        # Convert plot to image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img)
    
    # Save as GIF
    gif_buf = BytesIO()
    frames[0].save(
        gif_buf,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=200,
        loop=0
    )
    gif_buf.seek(0)
    return base64.b64encode(gif_buf.getvalue()).decode()

@app.get("/")
async def read_root():
    return {"message": "XAI-Phy API is running", "status": "ok"}

AVAILABLE_MODELS = ["mock", "gpt2", "bert", "distilbert", "sentence-transformer"]

@app.get("/api/models")
async def list_models():
    return {
        "models": AVAILABLE_MODELS,
        "current_model": current_model
    }

@app.post("/api/models/{model_name}")
async def set_model(model_name: str):
    global current_model, current_backend
    try:
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Invalid model name. Available models: {', '.join(AVAILABLE_MODELS)}")
        
        new_backend = model_manager.load_model(model_name)
        if new_backend is None:
            raise ValueError(f"Failed to load model {model_name}")
        
        current_model = model_name
        current_backend = new_backend
        return {"status": "success", "message": f"Model switched to {model_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/analyze")
async def analyze_text(text: str):
    try:
        if not text:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Text input is required"}
            )
        
        if current_backend is None:
            raise HTTPException(status_code=500, detail="No model is currently loaded")
        
        # Process text with current model
        try:
            model_result = await current_backend.process_text(text)
            tokens = model_result.get('tokens', [])
            attention_matrix = model_result.get('attention', None)
            embeddings = model_result.get('embeddings', None)
            
            if not tokens or attention_matrix is None or not isinstance(attention_matrix, np.ndarray):
                raise ValueError("Invalid model output: missing tokens or attention matrix")
            
            if embeddings is not None and not isinstance(embeddings, np.ndarray):
                embeddings = None

            # Generate visualizations
            attention_map = generate_attention_visualization(
                tokens=tokens,
                attention_matrix=attention_matrix,
                title=f"Attention Map ({current_model.capitalize()})"
            )
            attention_gif = generate_attention_animation(
                tokens=tokens,
                attention_matrix=attention_matrix
            )

            # Analyze text for integrity
            analysis = await current_backend.analyze_integrity(text, tokens, attention_matrix)
            
        except Exception as e:
            print(f"Processing error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Failed to process text: {str(e)}",
                    "model": current_model,
                    "attention_map": None,
                    "attention_gif": None,
                    "tokens": [],
                    "attention_matrix": None,
                    "embeddings": None,
                    "integrity_report": {
                        "hallucinations": [],
                        "contradictions": [],
                        "logical_consistency": []
                    }
                }
            )
        
        return JSONResponse(
            content={
                "status": "success",
                "model": current_model,
                "attention_map": attention_map,
                "attention_gif": attention_gif,
                "tokens": tokens,
                "attention_matrix": attention_matrix.tolist() if attention_matrix is not None else None,
                "embeddings": embeddings.tolist() if embeddings is not None else None,
                "integrity_report": analysis
            }
        )
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Failed to analyze text: {str(e)}",
                "model": current_model,
                "attention_map": None,
                "attention_gif": None,
                "tokens": [],
                "attention_matrix": None,
                "embeddings": None,
                "integrity_report": {
                    "hallucinations": [],
                    "contradictions": [],
                    "logical_consistency": []
                }
            }
        )

def generate_mock_attention_map():
    try:
        # Create a more interesting heatmap
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(Z, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Attention Score')
        plt.title('Attention Map')
        plt.axis('off')
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error generating attention map: {e}")
        raise

def generate_mock_attention_gif():
    try:
        # Create a series of more interesting frames
        size = (200, 200)
        frames = []
        
        # Generate 10 frames with moving patterns
        for i in range(10):
            # Create gradient pattern
            array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
            for x in range(size[0]):
                for y in range(size[1]):
                    r = int(255 * (np.sin(x/10 + i/3) + 1) / 2)
                    g = int(255 * (np.cos(y/10 - i/3) + 1) / 2)
                    b = int(255 * (np.sin((x+y)/20 + i/3) + 1) / 2)
                    array[x, y] = [r, g, b]
            
            img = Image.fromarray(array)
            frames.append(img)
        
        # Save to base64
        buf = io.BytesIO()
        frames[0].save(
            buf,
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=100,  # Faster animation
            loop=0
        )
        
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error generating attention gif: {e}")
        raise