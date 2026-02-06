/*
 * Arduino Uno Iris Classification Benchmark
 * Uses 4-layer SA-NN model with PROGMEM for memory efficiency
 */

#include "iris_model.h"

void setup() {
  Serial.begin(9600);
  while (!Serial); 
  
  Serial.println("Hello");
  Serial.println("Arduino Uno Iris Classification Benchmark");
  Serial.println("========================================");
  
  // Test setosa (class 0)
  Serial.println("\\nTesting setosa (class 0):");
  int8_t input_setosa[4] = {51, 35, 14, 2}; // Scaled by 10: [5.1, 3.5, 1.4, 0.2]
  int16_t output_setosa[3];
  interfere(input_setosa, output_setosa);
  
  Serial.print("Input: [5.1, 3.5, 1.4, 0.2] -> ");
  int max_idx = 0;
  if (output_setosa[1] > output_setosa[max_idx]) max_idx = 1;
  if (output_setosa[2] > output_setosa[max_idx]) max_idx = 2;
  
  if (max_idx == 0) Serial.println("SETOSA");
  else if (max_idx == 1) Serial.println("VERSICOLOR"); 
  else Serial.println("VIRGINICA");
  
  Serial.print("Output: [");
  Serial.print(output_setosa[0]);
  Serial.print(", ");
  Serial.print(output_setosa[1]);
  Serial.print(", ");
  Serial.println(output_setosa[2]);
  
  // Test versicolor (class 1)
  Serial.println("\n Testing versicolor (class 1):");
  int8_t input_versicolor[4] = {70, 32, 47, 14}; // Scaled by 10: [7.0, 3.2, 4.7, 1.4]
  int16_t output_versicolor[3];
  interfere(input_versicolor, output_versicolor);
  
  Serial.print("Input: [7.0, 3.2, 4.7, 1.4] -> ");
  max_idx = 0;
  if (output_versicolor[1] > output_versicolor[max_idx]) max_idx = 1;
  if (output_versicolor[2] > output_versicolor[max_idx]) max_idx = 2;
  
  if (max_idx == 0) Serial.println("SETOSA");
  else if (max_idx == 1) Serial.println("VERSICOLOR"); 
  else Serial.println("VIRGINICA");
  
  Serial.print("Output: [");
  Serial.print(output_versicolor[0]);
  Serial.print(", ");
  Serial.print(output_versicolor[1]);
  Serial.print(", ");
  Serial.println(output_versicolor[2]);
  
  // Test virginica (class 2)
  Serial.println("\nTesting virginica (class 2):");
  int8_t input_virginica[4] = {63, 33, 60, 25}; // Scaled by 10: [6.3, 3.3, 6.0, 2.5]
  int16_t output_virginica[3];
  interfere(input_virginica, output_virginica);
  
  Serial.print("Input: [6.3, 3.3, 6.0, 2.5] -> ");
  max_idx = 0;
  if (output_virginica[1] > output_virginica[max_idx]) max_idx = 1;
  if (output_virginica[2] > output_virginica[max_idx]) max_idx = 2;
  
  if (max_idx == 0) Serial.println("SETOSA");
  else if (max_idx == 1) Serial.println("VERSICOLOR"); 
  else Serial.println("VIRGINICA");
  
  Serial.print("Output: [");
  Serial.print(output_virginica[0]);
  Serial.print(", ");
  Serial.print(output_virginica[1]);
  Serial.print(", ");
  Serial.println(output_virginica[2]);
  
  Serial.println("\nIris classification benchmark complete!");
}

void loop() {
  // Nothing to do in the loop
}