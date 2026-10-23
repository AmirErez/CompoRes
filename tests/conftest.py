import matplotlib

# The tests save figures to files only. A window-based backend, the default on macOS, would open the figures in windows
# and make `plt.show()` wait for them to be closed.
matplotlib.use("Agg")
