"""
Utility functions for the simulator.
"""
import os


def get_output_dir():
    """
    Get the output directory for plots and results.
    Creates it if it doesn't exist.
    
    Returns:
    --------
    str : Absolute path to output directory
    """
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'output'
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_plot(filename, dpi=150):
    """
    Save current matplotlib plot to output directory.
    
    Parameters:
    -----------
    filename : str
        Name of the output file (e.g., 'my_plot.png')
    dpi : int
        Resolution in dots per inch
        
    Returns:
    --------
    str : Full path where file was saved
    """
    import matplotlib.pyplot as plt
    
    output_dir = get_output_dir()
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=dpi)
    return filepath
