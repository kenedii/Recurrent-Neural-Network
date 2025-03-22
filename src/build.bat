@echo off
echo Compiling matrix.c...
gcc -c matrix.c -o matrix.o -I include
if %ERRORLEVEL% NEQ 0 (
    echo Error compiling matrix.c
    exit /b %ERRORLEVEL%
)
echo Compiling backprop.c...
gcc -c backprop.c -o backprop.o -I include
if %ERRORLEVEL% NEQ 0 (
    echo Error compiling backprop.c
    exit /b %ERRORLEVEL%
)
echo Compiling RNN.c...
gcc -c RNN.c -o RNN.o -I include
if %ERRORLEVEL% NEQ 0 (
    echo Error compiling RNN.c
    exit /b %ERRORLEVEL%
)
echo Linking DLL...
gcc -shared -o rnn.dll matrix.o backprop.o RNN.o -lm
if %ERRORLEVEL% NEQ 0 (
    echo Error linking DLL
    exit /b %ERRORLEVEL%
)
echo Done.
pause