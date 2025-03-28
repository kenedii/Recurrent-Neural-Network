#!/bin/bash

echo "Compiling matrix.c with -fPIC..."
gcc -fPIC -c matrix.c -o matrix.o -I include
if [ $? -ne 0 ]; then
    echo "Error compiling matrix.c"
    exit 1
fi

echo "Compiling backprop.c with -fPIC..."
gcc -fPIC -c backprop.c -o backprop.o -I include
if [ $? -ne 0 ]; then
    echo "Error compiling backprop.c"
    exit 1
fi

echo "Compiling RNN_linux.c with -fPIC..."
gcc -fPIC -c RNN_linux.c -o RNN.o -I include
if [ $? -ne 0 ]; then
    echo "Error compiling RNN_linux.c"
    exit 1
fi

echo "Linking shared object..."
gcc -shared -o rnn.so matrix.o backprop.o RNN.o -lm
if [ $? -ne 0 ]; then
    echo "Error linking shared object"
    exit 1
fi

echo "Done."