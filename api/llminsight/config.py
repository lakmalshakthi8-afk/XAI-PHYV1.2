from dataclasses import dataclass, field

@dataclass
class PhysicsConfig:
    """Configuration for physics simulation."""
    steps: int = 50
    dt: float = 0.05
    C_semantic: float = 0.5
    C_attention: float = 0.5
    gamma_drag: float = 0.1
    attraction_threshold: float = 0.55
    repulsion_threshold: float = 0.45

@dataclass
class VizConfig:
    """Configuration for visualization."""
    figsize: tuple = (10, 8)
    attention_threshold: float = 0.05
    attention_alpha_scale: float = 2.0
    attention_linewidth_scale: float = 5.0
    create_gif: bool = True
    gif_fps: int = 10

@dataclass
class RuntimeConfig:
    """Configuration for runtime settings."""
    workdir: str = "outputs"

@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)