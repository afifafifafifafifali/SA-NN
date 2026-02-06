# SA-NN 
SA-NN (Shift Accumulated Neural Network) is a lightweight, fixed-point neural network inference
system designed for deployment on ultra-constrained microcontrollers. By replacing conventional
multiplications with bit-shift operations and accumulating the scaled inputs, SA-NN achieves nearfloating-point accuracy while minimizing computational complexity and memory footprint. The
user can pass in the PyTorch model and generate a header-file which they can plug-and-play. This
makes it suitable for microcontrollers with as little as 8-KB of RAM, such as Arduino Uno, ESP32
etc
## Intro
Trying to deploy neural-networks on constrained microcontrollers is challanging due to small amount of
RAM, lack of hardware multipliers for large numbers,
and absence of the Floating Point Units(FPU). SA-NN
(Shift-Accumulated Neural Network) is a neural-network
inference system which is designed to use the Non-volatile
program memory (PROGMEM) and replace multiplications with bit-shifts and accumulations. Enabling fixedpoint inference on devices with as little as 2-KB of RAM.
(Currently, only Arduino is supported for the program
memory feature)


## Usage
Usage is currently in USAGE.md

## Benchmarks

A benchmark on a arduino and a computer will be conducted. Will be avialiable soon!
