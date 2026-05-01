"""End-to-end pipeline: train RF/LR, export figures, write tables."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from train_model import train
from plot_figures import plot_all
from make_tables import make_all_tables

if __name__ == "__main__":
    train()
    plot_all()
    make_all_tables()
