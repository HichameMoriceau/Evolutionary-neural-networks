#-------------------------------------------------
#
# Project created by QtCreator 2016-01-30T13:59:09
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = evolutionary_nets
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    neuralnet.cpp \
    data_set.cpp \
    trainer.cpp \
    net_benchmark.cpp \
    evolutionary_trainer.cpp

HEADERS += \
    neuralnet.h \
    data_set.h \
    trainer.h \
    net_benchmark.h \
    evolutionary_trainer.h

# arguments for linear algebra library on linux
LIBS += -llapack -lblas -larmadillo

QMAKE_CXXFLAGS += -std=c++11

# armadillo library likes 02
QMAKE_CXXFLAGS_DEBUG -= -O
QMAKE_CXXFLAGS_DEBUG -= -O1
QMAKE_CXXFLAGS_DEBUG -= -O2
QMAKE_CXXFLAGS_DEBUG -= -O3
QMAKE_CXXFLAGS_DEBUG += -O2

# using OpenMP for faster processing
QMAKE_CXXFLAGS_DEBUG += -fopenmp
QMAKE_LFLAGS += -fopenmp

