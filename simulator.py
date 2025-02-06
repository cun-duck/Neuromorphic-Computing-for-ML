import torch
import numpy as np
from bindsnet.network import Network
from bindsnet.network.nodes import LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor

class NeuromorphicSimulator:
    def __init__(self, num_neurons=100, simulation_time=100):
        self.network = Network()
        self.num_neurons = num_neurons
        self.simulation_time = simulation_time
        
        # Buat layer neuron LIF
        self.input_layer = LIFNodes(n=num_neurons)
        self.output_layer = LIFNodes(n=num_neurons//2)
        
        # Hubungkan layer dengan synapse acak
        self.connection = Connection(
            source=self.input_layer, 
            target=self.output_layer, 
            w=torch.rand(num_neurons, num_neurons//2)
        )
        
        # Tambahkan monitor untuk merekam aktivitas
        self.monitor = Monitor(
            obj=self.output_layer,
            state_vars=["s", "v"],
            time=simulation_time
        )
        
        self.network.add_layer(self.input_layer, name="input")
        self.network.add_layer(self.output_layer, name="output")
        self.network.add_connection(self.connection, source="input", target="output")
        self.network.add_monitor(self.monitor, name="output_monitor")
    
    def run(self, input_spikes):
        # Konversi input ke spike trains
        spikes = torch.from_numpy(input_spikes).float()
        
        # Simulasi
        self.network.run(
            inputs={"input": spikes}, 
            time=self.simulation_time,
            progress_bar=True
        )
        
        return self.monitor.get("s"), self.monitor.get("v")