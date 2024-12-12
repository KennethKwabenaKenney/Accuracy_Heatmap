# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:20:17 2024

@author: kenneyke
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the Excel file
file_path = r'D:\ODOT_SPR866\My Label Data Work\New Manual Labelling\6_Analysis\RF_Site-Site_Cumulative_Accuracy_Matrix-Weighted-Final.xlsx'
df = pd.read_excel(file_path, header=None)

# Extract the directory path
directory = os.path.dirname(file_path)

# Extract the relevant part of the DataFrame (excluding headers)
accuracy_matrix = df.iloc[1:, 1:].astype(float)

# Set up the labels for rows and columns
labels = df.iloc[1:, 0].values
numeric_labels = [int(label.split()[-1]) for label in labels]

# # Create the heatmap
# plt.figure(figsize=(12, 10))
# ax = sns.heatmap(accuracy_matrix, annot=False, cmap="RdYlGn", xticklabels=numeric_labels, yticklabels=numeric_labels)

# # Customize the color bar to show highest, medium, and lowest accuracy values
# cbar = ax.collections[0].colorbar
# cbar.set_label('Accuracy', fontweight='bold', fontsize=14)

# # Get the highest, medium, and lowest accuracy values
# min_val = np.min(accuracy_matrix)
# max_val = np.max(accuracy_matrix)
# mid_val = (min_val + max_val) / 2

# cbar.set_ticks([min_val, mid_val, max_val])
# cbar.set_ticklabels([f'Lowest ({min_val:.2f})', f'Medium ({mid_val:.2f})', f'Highest ({max_val:.2f})'], fontweight='bold', fontsize=12)

# # Set the x-axis labels to be on top
# ax.xaxis.set_label_position('top')
# ax.xaxis.tick_top()

# # Set the title and labels
# plt.title('NN Cross Training-Testing Matrix (Site-to-Site)', pad=20, fontweight='bold', fontsize=16)
# plt.xlabel('Testing Sites', fontweight='bold', fontsize=14)
# plt.ylabel('Training Sites', fontweight='bold', fontsize=14)

# # Save the heatmap as an image in the same directory as the Excel file
# heatmap_file = os.path.join(directory, 'NN Cross Training-Testing Matrix_site-site_final.png')
# plt.savefig(heatmap_file, dpi=600, bbox_inches='tight')

# # Show the plot
# plt.show()

# Create the heatmap
plt.figure(figsize=(12, 10))
ax = sns.heatmap(accuracy_matrix, annot=False, cmap="RdYlGn", vmin=0, vmax=1, xticklabels=numeric_labels, yticklabels=numeric_labels)

# Customize the color bar to show a range from 0 to 1, with additional ticks for min, mid, and max values
cbar = ax.collections[0].colorbar
cbar.set_label('Accuracy', fontweight='bold', fontsize=14)

# Get the minimum, maximum, and average accuracy values
min_val = np.min(accuracy_matrix)
max_val = np.max(accuracy_matrix)
mid_val = (min_val + max_val) / 2

# Set ticks including the general range and specific min, mid, max values
cbar.set_ticks([0.00, min_val, 0.50, mid_val, 1.00, max_val])
cbar.set_ticklabels([
    '0', 
    f'Min ({min_val:.2f})', 
    '0.5', 
    f'Avg ({mid_val:.2f})', 
    '1', 
    f'Max ({max_val:.2f})'
], fontweight='bold', fontsize=12)

# Set the x-axis labels to be on top
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

# Set the title and labels
plt.title('RF Cross Training-Testing Matrix (Site-to-Site)', pad=20, fontweight='bold', fontsize=16)
plt.xlabel('Testing Sites', fontweight='bold', fontsize=14)
plt.ylabel('Training Sites', fontweight='bold', fontsize=14)

# Save the heatmap as an image in the same directory as the Excel file
heatmap_file = os.path.join(directory, 'RF_Site-Site_Cumulative_Accuracy_Matrix-Weighted-Final.png')
plt.savefig(heatmap_file, dpi=600, bbox_inches='tight')

# Show the plot
plt.show()
