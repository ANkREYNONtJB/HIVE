class TopologyAnalyzer:
    """
    Analysis tools for quantum topology shifts
    """
    def __init__(
        self,
        system: UnifiedResonanceSystem,
        constants: Optional[ResonanceConstants] = None
    ):
        self.system = system
        self.constants = constants or ResonanceConstants()
        
    def compute_topology_metrics(
        self,
        state: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute metrics for topology analysis"""
        # Extract relevant states
        field = state['field_states']['integrated_field']
        geometry = state['geometry_states']['final_geometry']
        loop = state['loop_states']['integrated_loop']
        
        # Compute field coherence
        field_coherence = torch.mean(torch.abs(field))
        
        # Compute geometry curvature
        geometry_grad = torch.gradient(geometry)[0]
        curvature = torch.mean(torch.abs(geometry_grad))
        
        # Compute loop complexity
        loop_fft = torch.fft.fft(loop)
        complexity = torch.mean(torch.abs(loop_fft))
        
        return {
            'field_coherence': field_coherence.item(),
            'geometry_curvature': curvature.item(),
            'loop_complexity': complexity.item()
        }
        
    def detect_topology_shifts(
        self,
        evolution: List[Dict[str, torch.Tensor]],
        threshold: float = 0.1
    ) -> List[Dict[str, Union[int, str, float]]]:
        """Detect significant topology shifts"""
        shifts = []
        metrics_history = []
        
        for i, state in enumerate(evolution):
            metrics = self.compute_topology_metrics(state)
            metrics_history.append(metrics)
            
            if i > 0:
                # Compute metric changes
                coherence_change = abs(
                    metrics['field_coherence'] -
                    metrics_history[-2]['field_coherence']
                )
                curvature_change = abs(
                    metrics['geometry_curvature'] -
                    metrics_history[-2]['geometry_curvature']
                )
                complexity_change = abs(
                    metrics['loop_complexity'] -
                    metrics_history[-2]['loop_complexity']
                )
                
                # Detect significant shifts
                if any(
                    change > threshold
                    for change in [
                        coherence_change,
                        curvature_change,
                        complexity_change
                    ]
                ):
                    shifts.append({
                        'time': i,
                        'coherence_change': coherence_change,
                        'curvature_change': curvature_change,
                        'complexity_change': complexity_change,
                        'total_magnitude': (
                            coherence_change +
                            curvature_change +
                            complexity_change
                        )
                    })
                    
        return shifts
        
    def analyze_resonance_stability(
        self,
        evolution: List[Dict[str, torch.Tensor]],
        window_size: int = 10
    ) -> Dict[str, np.ndarray]:
        """Analyze stability of resonance patterns"""
        metrics_history = []
        
        for state in evolution:
            metrics = self.compute_topology_metrics(state)
            metrics_history.append(metrics)
            
        # Convert to numpy arrays
        metrics_array = {
            key: np.array([m[key] for m in metrics_history])
            for key in metrics_history[0].keys()
        }
        
        # Compute stability metrics
        stability = {}
        for key, values in metrics_array.items():
            # Moving average
            ma = np.convolve(
                values,
                np.ones(window_size)/window_size,
                mode='valid'
            )
            
            # Moving standard deviation
            mstd = np.array([
                values[i:i+window_size].std()
                for i in range(len(values)-window_size+1)
            ])
            
            stability[f'{key}_ma'] = ma
            stability[f'{key}_mstd'] = mstd
            
        return stability