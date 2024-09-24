import os
import sys
import numpy as np
import tkinter as tk
from tkinter import Canvas, Scrollbar

class HopfieldNetwork:
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size[0] * pattern_size[1]
        self.weights = np.zeros((self.pattern_size, self.pattern_size))
        self.bias = np.zeros((1, self.pattern_size))
    
    def train(self, patterns):
        for pattern in patterns:
            pattern_flat = np.array(pattern).flatten()
            weight_update = np.outer(pattern_flat, pattern_flat)
            np.fill_diagonal(weight_update, 0)
            self.weights += weight_update / self.pattern_size
        self.bias = np.sum(self.weights, axis=0, keepdims=True)
        self.bias = self.bias.flatten()
        
    def recall(self, input_pattern, max_iterations=100):
        input_pattern_flat = np.array(input_pattern).flatten()
        for _ in range(max_iterations):
            output = np.sign(np.dot(self.weights, input_pattern_flat) - self.bias)
            
            for i in range(output.shape[0]):
                    if output[i] == 0:
                        output[i] = input_pattern_flat[i]

            if np.array_equal(output, input_pattern_flat):
                break
            input_pattern_flat = output
        return output.reshape(input_pattern.shape)

def display_result(training_patterns, test_patterns, result_patterns, pattern_size):
    root = tk.Tk()
    root.title("Hopfield Network Results")
    bigger = 25
    width1 = bigger * (pattern_size[1] + 1)
    height1 = bigger * (pattern_size[0] + 1)
    
    canvas_frame = tk.Frame(root)
    canvas_frame.pack(side=tk.LEFT)

    scrollbar = Scrollbar(canvas_frame, orient="vertical")
    scrollbar.pack(side="right", fill="y")

    canvas = Canvas(canvas_frame, yscrollcommand=scrollbar.set, width=800, height=800)
    canvas.pack(side="left", expand=True)
    
    scrollbar.config(command=canvas.yview)
    
    frame_container = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame_container, anchor='nw')
    
    for i, (train, test, result) in enumerate(zip(training_patterns, test_patterns, result_patterns)):
        frame = tk.Frame(frame_container, width=800, height=height1)
        frame.grid(row=i, column=0)
        
        canvas0 = Canvas(frame, width=width1, height=height1)
        canvas0.create_bitmap((150, 150), bitmap="gray12")
        for r in range(pattern_size[0]):
            for c in range(pattern_size[1]):
                if train[r, c] == 1:
                    canvas0.create_rectangle(c * bigger, r * bigger, (c + 1) * bigger, (r + 1) * bigger, fill="black")
                else:
                    canvas0.create_rectangle(c * bigger, r * bigger, (c + 1) * bigger, (r + 1) * bigger, fill="green")

        canvas1 = Canvas(frame, width=width1, height=height1)
        canvas1.create_bitmap((150, 150), bitmap="gray12")
        for r in range(pattern_size[0]):
            for c in range(pattern_size[1]):
                if test[r, c] == 1:
                    canvas1.create_rectangle(c * bigger, r * bigger, (c + 1) * bigger, (r + 1) * bigger, fill="black")
                else:
                    canvas1.create_rectangle(c * bigger, r * bigger, (c + 1) * bigger, (r + 1) * bigger, fill="green")

        canvas2 = Canvas(frame, width=width1, height=height1)
        canvas2.create_bitmap((150, 150), bitmap="gray12")
        for r in range(pattern_size[0]):
            for c in range(pattern_size[1]):
                if result[r, c] == 1:
                    canvas2.create_rectangle(c * bigger, r * bigger, (c + 1) * bigger, (r + 1) * bigger, fill="black")
                else:
                    canvas2.create_rectangle(c * bigger, r * bigger, (c + 1) * bigger, (r + 1) * bigger, fill="green")

        canvas0.grid(row=0, column=0)
        canvas1.grid(row=0, column=1)
        canvas2.grid(row=0, column=2)
        
    canvas.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))
    
    root.mainloop()

if __name__ == "__main__":
    # Example training and testing patterns
    if getattr(sys, 'frozen', False):
        current_dir = os.path.dirname(sys.executable)
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
    training_patterns = []
    training_pattern = []
    with open(os.path.join(current_dir, 'Basic_Training.txt'), 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if not((i + 1) % 13):
                continue
            
            training_line = []
            for c in line:
                if c == ' ':
                    training_line.append(-1)
                elif c == '1':
                    training_line.append(1)
            training_pattern.append(training_line)
            
            if (i + 1) % 13 == 12:
                training_patterns.append(np.array(training_pattern))
                training_pattern = []
            
    test_patterns = []
    test_pattern = []
    with open(os.path.join(current_dir, 'Basic_Testing.txt'), 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if not((i + 1) % 13):
                continue
            
            test_line = []
            for c in line:
                if c == ' ':
                    test_line.append(-1)
                elif c == '1':
                    test_line.append(1)
            test_pattern.append(test_line)
            
            if (i + 1) % 13 == 12:
                test_patterns.append(np.array(test_pattern))
                test_pattern = []
    # Train the Hopfield network
    pattern_size = [12, 9]
    hopfield = HopfieldNetwork(pattern_size)
    hopfield.train(training_patterns)

    # Test the Hopfield network
    result_patterns = [hopfield.recall(test_pattern, max_iterations=100) for test_pattern in test_patterns]

    # Display results in GUI
    display_result(training_patterns, test_patterns, result_patterns, pattern_size)
