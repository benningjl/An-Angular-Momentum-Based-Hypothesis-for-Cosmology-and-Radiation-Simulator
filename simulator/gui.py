#!/usr/bin/env python3
"""
Lightweight GUI for Angular Momentum Framework Simulator

A simple graphical interface for running simulations and viewing results.
"""

# Set matplotlib backend before any imports
import os
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent plot windows

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import sys
from io import StringIO
import subprocess
from pathlib import Path

# Add simulator to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class SimulatorGUI:
    """Main GUI application for the simulator."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Angular Momentum Framework Simulator")
        self.root.geometry("1200x800")
        
        # Styling
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # State variables
        self.running = False
        self.current_process = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Angular Momentum Framework Simulator",
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel - Simulation options
        self.setup_left_panel(main_frame)
        
        # Right panel - Output display
        self.setup_right_panel(main_frame)
        
        # Status bar
        self.setup_status_bar(main_frame)
        
    def setup_left_panel(self, parent):
        """Set up the left panel with simulation options."""
        
        left_frame = ttk.LabelFrame(parent, text="Simulations", padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Simulation buttons
        simulations = [
            ("Run ALL Tests", self.run_all_tests, "Run complete test suite"),
            ("Solar System", self.run_solar_system, "Simulate inner planets"),
            ("Binary Pulsar", self.run_binary_pulsar, "Orbital decay simulation"),
            ("Quantum Phenomena", self.run_quantum_tests, "Neutrinos, Bell, coherence"),
            ("Core Physics", self.run_core_physics, "Coupling and forces"),
            ("Black Holes", self.run_black_hole, "Minimum mass prediction"),
            ("Primordial Sphere", self.run_primordial, "Structure formation"),
            ("Galaxy Rotation", self.run_galaxy, "Rotation curves"),
            ("Predictions Summary", self.run_predictions, "All testable predictions"),
        ]
        
        for idx, (label, command, tooltip) in enumerate(simulations):
            btn = ttk.Button(
                left_frame,
                text=label,
                command=command,
                width=25
            )
            btn.grid(row=idx, column=0, pady=5, sticky=(tk.W, tk.E))
            self.create_tooltip(btn, tooltip)
        
        # Separator
        ttk.Separator(left_frame, orient='horizontal').grid(
            row=len(simulations), column=0, sticky=(tk.W, tk.E), pady=10
        )
        
        # Utility buttons
        ttk.Button(
            left_frame,
            text="Open Output Folder",
            command=self.open_output_folder,
            width=25
        ).grid(row=len(simulations)+1, column=0, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Button(
            left_frame,
            text="Clear Output",
            command=self.clear_output,
            width=25
        ).grid(row=len(simulations)+2, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Stop button
        self.stop_btn = ttk.Button(
            left_frame,
            text="Stop Simulation",
            command=self.stop_simulation,
            width=25,
            state='disabled'
        )
        self.stop_btn.grid(row=len(simulations)+3, column=0, pady=5, sticky=(tk.W, tk.E))
        
    def setup_right_panel(self, parent):
        """Set up the right panel with output display."""
        
        right_frame = ttk.LabelFrame(parent, text="Output", padding="10")
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        
        # Output text area with scrollbar
        self.output_text = scrolledtext.ScrolledText(
            right_frame,
            wrap=tk.WORD,
            width=80,
            height=35,
            font=('Consolas', 9),
            bg='#1e1e1e',
            fg='#d4d4d4',
            insertbackground='white'
        )
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure tags for colored output
        self.output_text.tag_configure('header', foreground='#4ec9b0', font=('Consolas', 10, 'bold'))
        self.output_text.tag_configure('success', foreground='#6a9955')
        self.output_text.tag_configure('error', foreground='#f48771')
        self.output_text.tag_configure('info', foreground='#9cdcfe')
        
        # Initial message
        self.write_output("Welcome to the Angular Momentum Framework Simulator!\n", 'header')
        self.write_output("Select a simulation from the left panel to begin.\n\n", 'info')
        self.write_output("This simulator tests predictions from the paper:\n")
        self.write_output("'An Angular-Momentum-Based Hypothesis for Cosmology and Radiation'\n\n")
        
    def setup_status_bar(self, parent):
        """Set up the status bar at the bottom."""
        
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=200)
        self.progress.pack(side=tk.RIGHT, padx=(10, 0))
        
    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget."""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = ttk.Label(tooltip, text=text, background="#ffffe0", relief=tk.SOLID, borderwidth=1)
            label.pack()
            widget.tooltip = tooltip
            
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                
        widget.bind('<Enter>', on_enter)
        widget.bind('<Leave>', on_leave)
        
    def write_output(self, text, tag=None):
        """Write text to the output area."""
        self.output_text.insert(tk.END, text, tag)
        self.output_text.see(tk.END)
        self.output_text.update()
        
    def clear_output(self):
        """Clear the output text area."""
        self.output_text.delete(1.0, tk.END)
        self.write_output("Output cleared.\n\n", 'info')
        
    def set_status(self, message, running=False):
        """Update the status bar."""
        self.status_label.config(text=message)
        self.running = running
        
        if running:
            self.progress.start(10)
            self.stop_btn.config(state='normal')
        else:
            self.progress.stop()
            self.stop_btn.config(state='disabled')
            
    def run_simulation(self, title, module_path, function_name=None):
        """Run a simulation in a separate thread."""
        
        if self.running:
            messagebox.showwarning("Simulation Running", "Please wait for the current simulation to complete.")
            return
            
        def run():
            self.set_status(f"Running: {title}", running=True)
            self.write_output(f"\n{'='*70}\n", 'header')
            self.write_output(f"{title.upper()}\n", 'header')
            self.write_output(f"{'='*70}\n\n", 'header')
            
            try:
                # Redirect stdout to capture output
                old_stdout = sys.stdout
                sys.stdout = output_capture = StringIO()
                
                # Import and run the simulation with proper namespace
                exec_globals = globals().copy()
                if function_name:
                    exec(f"from {module_path} import {function_name}", exec_globals)
                    exec(f"{function_name}()", exec_globals)
                else:
                    exec(f"import {module_path}", exec_globals)
                    exec(f"{module_path}.main()", exec_globals)
                
                # Get captured output
                output = output_capture.getvalue()
                sys.stdout = old_stdout
                
                # Display output
                self.write_output(output)
                self.write_output(f"\n[SUCCESS] {title} completed successfully!\n", 'success')
                
            except Exception as e:
                sys.stdout = old_stdout
                self.write_output(f"\n[ERROR] {str(e)}\n", 'error')
                import traceback
                self.write_output(traceback.format_exc(), 'error')
                
            finally:
                self.set_status("Ready", running=False)
                
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        
    def run_all_tests(self):
        """Run all tests."""
        self.run_simulation("Complete Test Suite", "run_all_tests", "main")
        
    def run_solar_system(self):
        """Run solar system simulation."""
        self.run_simulation("Solar System", "examples.solar_system", "main")
        
    def run_binary_pulsar(self):
        """Run binary pulsar simulation."""
        self.run_simulation("Binary Pulsar", "examples.binary_pulsar", "main")
        
    def run_quantum_tests(self):
        """Run quantum phenomena tests."""
        self.run_simulation("Quantum Phenomena", "examples.quantum_tests", "main")
        
    def run_core_physics(self):
        """Test core physics."""
        def run():
            self.set_status("Running: Core Physics Tests", running=True)
            self.write_output(f"\n{'='*70}\n", 'header')
            self.write_output("CORE PHYSICS TESTS\n", 'header')
            self.write_output(f"{'='*70}\n\n", 'header')
            
            try:
                from core import coupling, forces
                from orbital import comparison
                
                old_stdout = sys.stdout
                sys.stdout = output_capture = StringIO()
                
                coupling.test_framework()
                forces.test_forces()
                comparison.test_equivalence_principle()
                
                output = output_capture.getvalue()
                sys.stdout = old_stdout
                
                self.write_output(output)
                self.write_output("\n[SUCCESS] Core physics tests completed!\n", 'success')
                
            except Exception as e:
                sys.stdout = old_stdout
                self.write_output(f"\n[ERROR] {str(e)}\n", 'error')
                
            finally:
                self.set_status("Ready", running=False)
                
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        
    def run_black_hole(self):
        """Run black hole predictions."""
        def run():
            self.set_status("Calculating black hole predictions", running=True)
            self.write_output(f"\n{'='*70}\n", 'header')
            self.write_output("BLACK HOLE MINIMUM MASS PREDICTION\n", 'header')
            self.write_output(f"{'='*70}\n\n", 'header')
            
            try:
                from cosmology import primordial
                
                old_stdout = sys.stdout
                sys.stdout = output_capture = StringIO()
                
                predictions.black_hole_minimum_mass()
                
                output = output_capture.getvalue()
                sys.stdout = old_stdout
                
                self.write_output(output)
                self.write_output("\n[SUCCESS] Black hole predictions completed!\n", 'success')
                
            except Exception as e:
                sys.stdout = old_stdout
                self.write_output(f"\n[ERROR] {str(e)}\n", 'error')
                
            finally:
                self.set_status("Ready", running=False)
                
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        
    def run_primordial(self):
        """Run primordial sphere simulation."""
        def run():
            self.set_status("Simulating primordial sphere", running=True)
            self.write_output(f"\n{'='*70}\n", 'header')
            self.write_output("PRIMORDIAL SPHERE & STRUCTURE FORMATION\n", 'header')
            self.write_output(f"{'='*70}\n\n", 'header')
            
            try:
                from cosmology import primordial
                
                old_stdout = sys.stdout
                sys.stdout = output_capture = StringIO()
                
                sphere = primordial.calculate_primordial_parameters()
                primordial.simulate_structure_formation()
                
                output = output_capture.getvalue()
                sys.stdout = old_stdout
                
                self.write_output(output)
                self.write_output("\n[SUCCESS] Primordial sphere simulation completed!\n", 'success')
                
            except Exception as e:
                sys.stdout = old_stdout
                self.write_output(f"\n[ERROR] {str(e)}\n", 'error')
                
            finally:
                self.set_status("Ready", running=False)
                
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        
    def run_galaxy(self):
        """Run galaxy rotation curve tests."""
        def run():
            self.set_status("Testing galaxy rotation curves", running=True)
            self.write_output(f"\n{'='*70}\n", 'header')
            self.write_output("GALAXY ROTATION CURVES\n", 'header')
            self.write_output(f"{'='*70}\n\n", 'header')
            
            try:
                from cosmology import galaxies
                
                old_stdout = sys.stdout
                sys.stdout = output_capture = StringIO()
                
                galaxies.test_rotation_curves()
                galaxies.generate_rotation_curve_plot()
                
                output = output_capture.getvalue()
                sys.stdout = old_stdout
                
                self.write_output(output)
                self.write_output("\n[SUCCESS] Galaxy rotation tests completed!\n", 'success')
                
            except Exception as e:
                sys.stdout = old_stdout
                self.write_output(f"\n[ERROR] {str(e)}\n", 'error')
                
            finally:
                self.set_status("Ready", running=False)
                
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        
    def run_predictions(self):
        """Show all testable predictions."""
        def run():
            self.set_status("Calculating all predictions", running=True)
            self.write_output(f"\n{'='*70}\n", 'header')
            self.write_output("TESTABLE PREDICTIONS SUMMARY\n", 'header')
            self.write_output(f"{'='*70}\n\n", 'header')
            
            try:
                from cosmology import predictions
                
                old_stdout = sys.stdout
                sys.stdout = output_capture = StringIO()
                
                predictions.summary_of_predictions()
                
                output = output_capture.getvalue()
                sys.stdout = old_stdout
                
                self.write_output(output)
                self.write_output("\n[SUCCESS] Predictions summary completed!\n", 'success')
                
            except Exception as e:
                sys.stdout = old_stdout
                self.write_output(f"\n[ERROR] {str(e)}\n", 'error')
                
            finally:
                self.set_status("Ready", running=False)
                
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        
    def open_output_folder(self):
        """Open the output folder in file explorer."""
        output_dir = Path(__file__).parent / 'output'
        output_dir.mkdir(exist_ok=True)
        
        if sys.platform == 'win32':
            os.startfile(output_dir)
        elif sys.platform == 'darwin':  # macOS
            subprocess.run(['open', output_dir])
        else:  # Linux
            subprocess.run(['xdg-open', output_dir])
            
        self.write_output(f"Opened output folder: {output_dir}\n", 'info')
        
    def stop_simulation(self):
        """Stop the current simulation."""
        # Note: This is a simplified stop - proper implementation would need
        # more sophisticated thread management
        self.set_status("Stopping...", running=False)
        self.write_output("\n[WARNING] Stop requested (simulation may complete current step)\n", 'info')


def main():
    """Launch the GUI."""
    root = tk.Tk()
    app = SimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
