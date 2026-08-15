import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
