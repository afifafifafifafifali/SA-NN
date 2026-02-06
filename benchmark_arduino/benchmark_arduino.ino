/*
 * Arduino Uno Sentiment Analysis Benchmark
 * Uses SA-NN model with PROGMEM for memory efficiency
 */

#include "sentiment_model.h"

int16_t output_buffer[2];
void setup() {
  Serial.begin(9600);
  while (!Serial); 
  
  Serial.println("Arduino Uno Sentiment Analysis Benchmark");
  Serial.println("=====================================");
  
  Serial.println("\n Testing positive sentiment:");
  interfere_text("this movie is great", output_buffer);
  Serial.print("Input: 'this movie is great' -> ");
  if (output_buffer[1] > output_buffer[0]) {
    Serial.println("POSITIVE");
  } else {
    Serial.println("NEGATIVE");
  }
  Serial.print("Output: [");
  Serial.print(output_buffer[0]);
  Serial.print(", ");
  Serial.println(output_buffer[1]);
  
  Serial.println("\n Testing negative sentiment:");
  interfere_text("terrible film hate it", output_buffer);
  Serial.print("Input: 'terrible film hate it' -> ");
  if (output_buffer[1] > output_buffer[0]) {
    Serial.println("POSITIVE");
  } else {
    Serial.println("NEGATIVE");
  }
  Serial.print("Output: [");
  Serial.print(output_buffer[0]);
  Serial.print(", ");
  Serial.println(output_buffer[1]);
  
  Serial.println("\n Testing mixed sentiment:");
  interfere_text("good movie but boring plot", output_buffer);
  Serial.print("Input: 'good movie but boring plot' -> ");
  if (output_buffer[1] > output_buffer[0]) {
    Serial.println("POSITIVE");
  } else {
    Serial.println("NEGATIVE");
  }
  Serial.print("Output: [");
  Serial.print(output_buffer[0]);
  Serial.print(", ");
  Serial.println(output_buffer[1]);
  
  Serial.println("\nBenchmark complete!");
}

void loop() {
  // Nothing to do in the loop
}


