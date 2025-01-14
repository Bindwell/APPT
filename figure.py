import matplotlib.pyplot as plt
import numpy as np

# Create data dictionary with OURS (APT) as the first entry
r_value = {
    'OURS': 0.79,  # Adding APT as the first entry
    'PPB-Affinity': 0.671,
    'Specific Geometry': 0.53,
    'ASA and I-RMSD': 0.31
}

# Create lists for plotting
methods = list(r_value.keys())
values = list(r_value.values())

# Create the plot
plt.figure(figsize=(10, 6))

# Create bars with different colors (first bar in blue, rest in muted red)
colors = ['#4A90E2'] + ['#E57373'] * (len(methods) - 1)
bars = plt.bar(methods, values, color=colors, width=0.5, edgecolor='black', linewidth=0.8)

# Customize the plot
plt.ylabel('R Value', fontsize=12)
plt.title('Comparison of R Value Across Different Methods', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)

# Set y-axis limits to make differences appear larger
plt.ylim(0.2, 0.79)  # Narrow the y-axis range to emphasize differences

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=10)

# Add a horizontal line at y=0 for reference
plt.axhline(0.2, color='gray', linewidth=0.8, linestyle='--')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()
