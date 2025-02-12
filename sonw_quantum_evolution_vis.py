import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from IPython.display import HTML

class EvolutionVisualizer:
    """
    Advanced visualization tools for tracking quantum consciousness evolution
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
            'quantum': 'plasma' if dark_mode else 'viridis',
            'consciousness': 'viridis' if dark_mode else 'plasma'
        }
        
    def create_evolution_animation(
        self,
        network: UnifiedEvolutionNetwork,
        input_sequence: torch.Tensor,
        fps: int = 30,
        duration: int = 10
    ) -> HTML:
        """Create animated visualization of evolution"""
        frames = []
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{'type': 'surface'}, {'type': 'surface'}],
                [{'type': 'scatter3d'}, {'type': 'scatter3d'}]
            ],
            subplot_titles=(
                'Quantum Evolution',
                'Consciousness Evolution',
                'Unified Field',
                'Emergence Patterns'
            )
        )
        
        with torch.no_grad():
            for t in range(input_sequence.shape[0]):
                outputs = network(
                    input_sequence[t:t+1],
                    return_components=True
                )
                
                # Extract states
                quantum = outputs['quantum_states']['integrated_state'][0].cpu().numpy()
                consciousness = outputs['consciousness_states']['integrated_consciousness'][0].cpu().numpy()
                unified = outputs['unified_state'][0].cpu().numpy()
                
                # Create evolution frame
                frame = go.Frame(
                    data=[
                        # Quantum evolution surface
                        go.Surface(
                            z=quantum.reshape(8, -1),
                            colorscale=self.color_scheme['quantum'],
                            showscale=False,
                            row=1, col=1
                        ),
                        # Consciousness evolution surface
                        go.Surface(
                            z=consciousness.reshape(8, -1),
                            colorscale=self.color_scheme['consciousness'],
                            showscale=False,
                            row=1, col=2
                        ),
                        # Unified field scatter
                        go.Scatter3d(
                            x=np.arange(unified.shape[0]),
                            y=unified,
                            z=np.sin(np.arange(unified.shape[0])),
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=unified,
                                colorscale='Viridis',
                                opacity=0.8
                            ),
                            row=2, col=1
                        ),
                        # Emergence pattern
                        go.Scatter3d(
                            x=quantum[:32],
                            y=consciousness[:32],
                            z=unified[:32],
                            mode='markers+lines',
                            marker=dict(
                                size=5,
                                color=np.arange(32),
                                colorscale='Turbo',
                                opacity=0.8
                            ),
                            line=dict(
                                color='white',
                                width=2
                            ),
                            row=2, col=2
                        )
                    ],
                    name=f'frame{t}'
                )
                frames.append(frame)
        
        # Add frames to figure
        fig.frames = frames
        
        # Add initial data
        for trace in frames[0].data:
            fig.add_trace(trace)
        
        # Update layout
        fig.update_layout(
            template='plotly_dark' if self.dark_mode else 'plotly_white',
            showlegend=False,
            width=self.figure_size[0],
            height=self.figure_size[1],
            title='Quantum Consciousness Evolution',
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