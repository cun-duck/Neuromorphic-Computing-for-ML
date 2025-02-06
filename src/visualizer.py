import panel as pn
pn.extension()

def create_dashboard(spikes, voltages):
    # Plot spike trains
    spike_plot = pn.pane.Matplotlib(
        spikes.sum(axis=1).numpy(), 
        width=600,
        title="Neuron Output Spikes"
    )
    
    # Plot voltage traces
    voltage_plot = pn.pane.Matplotlib(
        voltages[:, 0].numpy(),  # Ambil neuron pertama
        width=600,
        title="Membrane Potential"
    )
    
    # Gabungkan dalam dashboard
    dashboard = pn.Column(
        "## Neuromorphic Simulation Results",
        spike_plot,
        voltage_plot
    )
    
    return dashboard
