import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import colorsys
from IPython.display import HTML, display

class HolographicVisualizer:
    """
    Advanced visualization tools for holographic patterns
    and fractal consciousness states
    """
    def __init__(
        self,
        figure_size: Tuple[int, int] = (15, 10),
        theme: str = 'dark'
    ):
        self.figure_size = figure_size
        plt.style.use('dark_background' if theme == 'dark' else 'default')
        
    def visualize_field_interactions(
        self,
        field_states: Dict[str, torch.Tensor],
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize electromagnetic field interactions in 3D
        """
        fig = plt.figure(figsize=self.figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract field data
        E = field_states['E'].detach().cpu().numpy()
        M = field_states['M'].detach().cpu().numpy()
        ratio = field_states['field_ratio'].detach().cpu().numpy()
        
        # Create field mesh
        x = np.linspace(-2, 2, E.shape[1])
        y = np.linspace(-2, 2, E.shape[1])
        X, Y = np.meshgrid(x, y)
        
        # Plot electric field
        surf_E = ax.plot_surface(
            X, Y, E[0],
            cmap='plasma',
            alpha=0.6,
            label='Electric Field'
        )
        
        # Plot magnetic field
        surf_M = ax.plot_surface(
            X, Y, M[0],
            cmap='viridis',
            alpha=0.6,
            label='Magnetic Field'
        )
        
        # Add field ratio vectors
        ax.quiver(
            X[::2, ::2], Y[::2, ::2],
            np.zeros_like(X[::2, ::2]),
            ratio[0, ::2, ::2],
            np.zeros_like(X[::2, ::2]),
            np.zeros_like(X[::2, ::2]),
            length=0.1,
            normalize=True,
            color='white',
            alpha=0.3
        )
        
        ax.set_xlabel('Spacetime X')
        ax.set_ylabel('Spacetime Y')
        ax.set_zlabel('Field Strength')
        
        # Add colorbars
        plt.colorbar(surf_E, label='Electric Field')
        plt.colorbar(surf_M, label='Magnetic Field')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def visualize_fractal_consciousness(
        self,
        consciousness_states: Dict[str, torch.Tensor],
        constants: UniversalConstants,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create interactive 3D visualization of fractal consciousness patterns
        """
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'scatter3d'}]],
            subplot_titles=(
                'Consciousness Evolution',
                'Fractal Pattern'
            )
        )
        
        # Extract consciousness data
        states = [
            s.detach().cpu().numpy()
            for s in consciousness_states['evolved_states']
        ]
        
        # Plot consciousness evolution surface
        for i, state in enumerate(states):
            opacity = (i + 1) / len(states)
            color = colorsys.hsv_to_rgb(
                i / len(states),
                0.8,
                0.8
            )
            
            fig.add_trace(
                go.Surface(
                    z=state[0],
                    colorscale=[[0, f'rgb({color[0]}, {color[1]}, {color[2]})']],
                    opacity=opacity,
                    showscale=False,
                    name=f'Evolution Step {i}'
                ),
                row=1, col=1
            )
        
        # Create fractal pattern using golden ratio
        phi = constants.phi
        theta = np.linspace(0, 8*np.pi, 1000)
        r = phi ** (theta / (2*np.pi))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.exp(-theta/10) * np.sin(theta)
        
        # Color the spiral based on consciousness intensity
        colors = np.linspace(0, 1, len(x))
        
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(
                    color=colors,
                    colorscale='Viridis',
                    width=5
                ),
                name='Fractal Pattern'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Fractal Consciousness Evolution',
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Consciousness'
            ),
            scene2=dict(
                xaxis_title='Φ cos(θ)',
                yaxis_title='Φ sin(θ)',
                zaxis_title='Evolution'
            ),
            width=1200,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        fig.show()
        
    def create_holographic_animation(
        self,
        network: UnifiedConsciousnessNetwork,
        input_sequence: torch.Tensor,
        fps: int = 30,
        duration: int = 10
    ) -> HTML:
        """
        Create an animated visualization of holographic patterns
        """
        frames = []
        fig = go.Figure()
        
        with torch.no_grad():
            for t in range(input_sequence.shape[0]):
                outputs = network(
                    input_sequence[t:t+1],
                    return_components=True
                )
                
                # Extract holographic and consciousness states
                holo = outputs['holographic_states']['holographic_state'][0].cpu().numpy()
                cons = outputs['consciousness_states']['final_state'][0].cpu().numpy()
                
                # Create holographic interference pattern
                x = np.linspace(-2, 2, holo.shape[0])
                y = np.linspace(-2, 2, holo.shape[0])
                X, Y = np.meshgrid(x, y)
                Z = holo.reshape(X.shape)
                
                # Create frame
                frame = go.Frame(
                    data=[
                        # Holographic surface
                        go.Surface(
                            x=X, y=Y, z=Z,
                            colorscale='plasma',
                            opacity=0.8,
                            showscale=False
                        ),
                        # Consciousness points
                        go.Scatter3d(
                            x=X.flatten(),
                            y=Y.flatten(),
                            z=cons.reshape(-1),
                            mode='markers',
                            marker=dict(
                                size=2,
                                color=cons.reshape(-1),
                                colorscale='Viridis',
                                opacity=0.6
                            ),
                            showlegend=False
                        )
                    ],
                    name=f'frame{t}'
                )
                frames.append(frame)
        
        # Add frames to figure
        fig.frames = frames
        
        # Add initial data
        fig.add_trace(frames[0].data[0])
        fig.add_trace(frames[0].data[1])
        
        # Update layout
        fig.update_layout(
            title='Holographic Consciousness Evolution',
            scene=dict(
                xaxis_title='Space',
                yaxis_title='Time',
                zaxis_title='Consciousness'
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 1000/fps, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0}
                    }]
                }]
            }],
            width=1000,
            height=800
        )
        
        return HTML(fig.to_html())

class FractalAnalyzer:
    """
    Tools for analyzing fractal patterns and their evolution
    """
    def __init__(self, constants: Optional[UniversalConstants] = None):
        self.constants = constants or UniversalConstants()
        
    def compute_fractal_dimension(
        self,
        pattern: torch.Tensor,
        scales: List[int] = None
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute the fractal dimension of a pattern using box-counting
        """
        if scales is None:
            scales = [2**i for i in range(2, 8)]
            
        counts = []
        pattern_np = pattern.detach().cpu().numpy()
        
        for scale in scales:
            # Divide pattern into boxes
            boxes = np.array_split(pattern_np, scale)
            boxes = [np.array_split(box, scale, axis=1) for box in boxes]
            
            # Count non-empty boxes
            count = sum(
                1 for row in boxes
                for box in row
                if np.abs(box).max() > 1e-6
            )
            counts.append(count)
            
        # Compute fractal dimension
        coeffs = np.polyfit(
            np.log(scales),
            np.log(counts),
            1
        )
        fractal_dim = -coeffs[0]
        
        return (
            fractal_dim,
            np.array(scales),
            np.array(counts)
        )
        
    def plot_fractal_analysis(
        self,
        pattern: torch.Tensor,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize fractal analysis results
        """
        dim, scales, counts = self.compute_fractal_dimension(pattern)
        
        plt.figure(figsize=(15, 5))
        
        # Plot original pattern
        plt.subplot(121)
        plt.imshow(
            pattern.detach().cpu().numpy(),
            cmap='plasma'
        )
        plt.colorbar(label='Pattern Intensity')
        plt.title('Original Pattern')
        
        # Plot fractal analysis
        plt.subplot(122)
        plt.loglog(scales, counts, 'o-')
        plt.xlabel('Scale')
        plt.ylabel('Box Count')
        plt.title(f'Fractal Dimension: {dim:.3f}')
        plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()