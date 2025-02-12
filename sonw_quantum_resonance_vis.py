import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from IPython.display import HTML
import colorsys

class ResonanceVisualizer:
    """
    Advanced visualization tools for quantum resonance patterns
    """
    def __init__(
        self,
        dark_mode: bool = True,
        figure_size: Tuple[int, int] = (1200, 800)
    ):
        self.dark_mode = dark_mode
        self.figure_size = figure_size
        self.color_scheme = {
            'background': 'rgb(17,17,17)' if dark_mode else 'rgb(255,255,255)',
            'text': 'rgb(255,255,255)' if dark_mode else 'rgb(17,17,17)',
            'epsilon': 'plasma',
            'omega': 'viridis',
            'loop': 'turbo'
        }
        
    def create_resonance_animation(
        self,
        system: UnifiedResonanceSystem,
        evolution: List[Dict[str, torch.Tensor]],
        fps: int = 30,
        duration: int = 10
    ) -> HTML:
        """Create animated visualization of resonance patterns"""
        frames = []
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{'type': 'surface'}, {'type': 'surface'}],
                [{'type': 'scatter3d'}, {'type': 'scatter3d'}]
            ],
            subplot_titles=(
                'Morphogenetic Field',
                'Quantum Geometry',
                'Strange Loop',
                'Unified Resonance'
            )
        )
        
        for t, state in enumerate(evolution):
            # Extract states
            field = state['field_states']['integrated_field'][0].cpu().numpy()
            geometry = state['geometry_states']['final_geometry'][0].cpu().numpy()
            loop = state['loop_states']['integrated_loop'][0].cpu().numpy()
            unified = state['unified_state'][0].cpu().numpy()
            
            # Create resonance frame
            frame = go.Frame(
                data=[
                    # Morphogenetic field surface
                    go.Surface(
                        z=field.reshape(8, -1),
                        colorscale=self.color_scheme['epsilon'],
                        showscale=False,
                        row=1, col=1
                    ),
                    # Quantum geometry surface
                    go.Surface(
                        z=geometry.reshape(8, -1),
                        colorscale=self.color_scheme['omega'],
                        showscale=False,
                        row=1, col=2
                    ),
                    # Strange loop trajectory
                    go.Scatter3d(
                        x=np.arange(loop.shape[0]),
                        y=loop,
                        z=np.sin(np.arange(loop.shape[0])) * loop,
                        mode='lines+markers',
                        marker=dict(
                            size=4,
                            color=loop,
                            colorscale=self.color_scheme['loop'],
                            opacity=0.8
                        ),
                        line=dict(
                            color='white',
                            width=2
                        ),
                        row=2, col=1
                    ),
                    # Unified resonance pattern
                    go.Scatter3d(
                        x=field[:32],
                        y=geometry[:32],
                        z=loop[:32],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=unified[:32],
                            colorscale='Turbo',
                            opacity=0.8
                        ),
                        row=2, col=2
                    )
                ],
                name=f'frame{t}'
            )
            frames.append(frame)
            
        # Update layout
        fig.update_layout(
            template='plotly_dark' if self.dark_mode else 'plotly_white',
            showlegend=False,
            width=self.figure_size[0],
            height=self.figure_size[1],
            title='Quantum Resonance Evolution',
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
            }]
        )
        
        return HTML(fig.to_html())