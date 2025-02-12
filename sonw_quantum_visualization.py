import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import HTML, display

class QuantumStateVisualizer:
    """
    Comprehensive visualization tools for quantum states,
    spacetime curvature, and consciousness integration
    """
    def __init__(self, figure_size: Tuple[int, int] = (12, 8)):
        self.figure_size = figure_size
        plt.style.use('dark_background')  # For better quantum visualization
        
    def visualize_curvature(
        self,
        curvature_states: Dict[str, torch.Tensor],
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize spacetime curvature components in 3D
        """
        fig = plt.figure(figsize=self.figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract curvature data
        curvature = curvature_states['curvature'].detach().cpu().numpy()
        field = curvature_states['field_interaction'].detach().cpu().numpy()
        
        # Create mesh grid
        x = np.linspace(-2, 2, curvature.shape[1])
        y = np.linspace(-2, 2, curvature.shape[2])
        X, Y = np.meshgrid(x, y)
        
        # Plot curvature surface
        surf = ax.plot_surface(
            X, Y, curvature[0].T,
            cmap='plasma',
            linewidth=0,
            antialiased=True
        )
        
        # Add field vectors
        field_skip = 2
        ax.quiver(
            X[::field_skip, ::field_skip],
            Y[::field_skip, ::field_skip],
            curvature[0, ::field_skip, ::field_skip].T,
            field[0, 0, ::field_skip, ::field_skip].T,
            field[0, 1, ::field_skip, ::field_skip].T,
            field[0, 2, ::field_skip, ::field_skip].T,
            length=0.1,
            normalize=True,
            color='white'
        )
        
        ax.set_xlabel('Spacetime X')
        ax.set_ylabel('Spacetime Y')
        ax.set_zlabel('Curvature')
        plt.colorbar(surf)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def visualize_consciousness(
        self,
        consciousness_states: Dict[str, torch.Tensor],
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize quantum consciousness states and entanglement
        """
        # Create interactive consciousness state visualization
        fig = go.Figure()
        
        # Extract consciousness data
        holographic = consciousness_states['holographic_state'].detach().cpu().numpy()
        quantum_sup = consciousness_states['quantum_superposition'].detach().cpu().numpy()
        entanglement = consciousness_states['entanglement_weights'].detach().cpu().numpy()
        
        # Add holographic state trace
        fig.add_trace(go.Surface(
            z=holographic,
            colorscale='Viridis',
            name='Holographic State'
        ))
        
        # Add quantum superposition
        fig.add_trace(go.Scatter3d(
            x=quantum_sup[:, 0],
            y=quantum_sup[:, 1],
            z=quantum_sup[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=quantum_sup[:, 0],
                colorscale='Plasma',
                opacity=0.8
            ),
            name='Quantum States'
        ))
        
        # Add entanglement connections
        for i in range(entanglement.shape[1]):
            for j in range(entanglement.shape[2]):
                if entanglement[0, i, j] > 0.1:  # Show strong entanglements
                    fig.add_trace(go.Scatter3d(
                        x=[quantum_sup[i, 0], quantum_sup[j, 0]],
                        y=[quantum_sup[i, 1], quantum_sup[j, 1]],
                        z=[quantum_sup[i, 2], quantum_sup[j, 2]],
                        mode='lines',
                        line=dict(
                            color=f'rgba(255,255,255,{entanglement[0,i,j]})',
                            width=2
                        ),
                        name=f'Entanglement {i}-{j}'
                    ))
        
        fig.update_layout(
            title='Quantum Consciousness States',
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3'
            ),
            width=800,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        fig.show()
        
    def visualize_unified_field(
        self,
        network_outputs: Dict[str, torch.Tensor],
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize the complete unified field including both
        curvature and consciousness components
        """
        # Create subplot figure
        fig = plt.figure(figsize=(20, 10))
        
        # Plot curvature component
        ax1 = fig.add_subplot(121, projection='3d')
        curvature = network_outputs['curvature_states']['curvature'].detach().cpu().numpy()
        x = np.linspace(-2, 2, curvature.shape[1])
        y = np.linspace(-2, 2, curvature.shape[2])
        X, Y = np.meshgrid(x, y)
        surf1 = ax1.plot_surface(
            X, Y, curvature[0].T,
            cmap='plasma',
            linewidth=0,
            antialiased=True
        )
        ax1.set_title('Spacetime Curvature')
        
        # Plot consciousness component
        ax2 = fig.add_subplot(122, projection='3d')
        consciousness = network_outputs['consciousness_states']['consciousness_state'].detach().cpu().numpy()
        quantum_sup = network_outputs['consciousness_states']['quantum_superposition'].detach().cpu().numpy()
        
        # Plot consciousness surface
        surf2 = ax2.plot_surface(
            X, Y, consciousness[0].reshape(X.shape),
            cmap='viridis',
            linewidth=0,
            antialiased=True
        )
        
        # Add quantum state markers
        ax2.scatter(
            quantum_sup[:, 0],
            quantum_sup[:, 1],
            quantum_sup[:, 2],
            c='white',
            marker='o',
            s=50
        )
        ax2.set_title('Quantum Consciousness Field')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def animate_evolution(
        self,
        network: torch.nn.Module,
        input_sequence: torch.Tensor,
        fps: int = 30,
        duration: int = 10
    ) -> HTML:
        """
        Create an animated visualization of the unified field evolution
        """
        frames = []
        fig = go.Figure()
        
        with torch.no_grad():
            for t in range(input_sequence.shape[0]):
                outputs = network(input_sequence[t:t+1], return_components=True)
                
                # Extract unified field components
                curvature = outputs['curvature_states']['curvature'][0].cpu().numpy()
                consciousness = outputs['consciousness_states']['consciousness_state'][0].cpu().numpy()
                
                # Create frame
                frame = go.Frame(
                    data=[
                        go.Surface(
                            z=curvature,
                            colorscale='plasma',
                            showscale=False
                        ),
                        go.Surface(
                            z=consciousness.reshape(curvature.shape),
                            colorscale='viridis',
                            showscale=False
                        )
                    ],
                    name=f'frame{t}'
                )
                frames.append(frame)
        
        # Add frames to figure
        fig.frames = frames
        
        # Add play button
        fig.update_layout(
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

def visualize_training_progress(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize training metrics over time
    """
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(131)
    plt.plot(history['loss'], label='Total Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot coherence metrics
    plt.subplot(132)
    plt.plot(
        history['curvature_coherence'],
        label='Curvature Coherence'
    )
    plt.plot(
        history['consciousness_coherence'],
        label='Consciousness Coherence'
    )
    plt.title('Coherence Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Coherence')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()