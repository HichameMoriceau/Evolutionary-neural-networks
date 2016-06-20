#!/bin/bash


(exec "./plotter.m" "data/results-bp-fixed-bcm.mat" "WINE-BP-FIXED")

(exec "./plotter.m" "data/results-bp-fixed-wine.mat" "WINE-BP-FIXED")

(exec "./plotter.m" "data/results-bp-fixed-bcr.mat" "BCR-BP-FIXED")

(exec "./plotter.m" "data/results-bp-fixed-iris.mat" "IRIS-BP-FIXED")


(exec "./plotter.m" "data/results-bp-cascade-bcm.mat" "BCM-BP-CASCADE")

(exec "./plotter.m" "data/results-bp-cascade-wine.mat" "WINE-BP-CASCADE")

(exec "./plotter.m" "data/results-bp-cascade-bcr.mat" "BCR-BP-CASCADE")

(exec "./plotter.m" "data/results-bp-cascade-iris.mat" "IRIS-BP-CASCADE")
