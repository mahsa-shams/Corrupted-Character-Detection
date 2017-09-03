QT += core
QT -= gui

CONFIG += c++11

TARGET = PlateCharacterDetection
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp

INCLUDEPATH += "/usr/local/include/"

LIBS += `pkg-config --libs opencv`
