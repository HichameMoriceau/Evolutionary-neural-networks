#!/bin/bash

(exec "./plotter.m" "data/results-neat-bcm.mat" "BCM-NEAT")

(exec "./plotter.m" "data/results-neat-iris.mat" "IRIS-NEAT")

(exec "./plotter.m" "data/results-neat-wine.mat" "WINE-NEAT")

#(exec "./plotter.m" "data/results-neat-bcr.mat" "BCR-NEAT")
