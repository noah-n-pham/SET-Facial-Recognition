/**
 * Face Recognition Hardware Controller
 * 
 * Receives serial commands from Jetson Nano and controls hardware.
 * Commands:
 *   - PERSON:<name>\n - Person recognized
 *   - COORD:<x>,<y>\n - Face center coordinates
 *   - UNKNOWN\n - Unknown person detected
 */

String inputBuffer = "";
const int LED_PIN = 13;  // Built-in LED (change for your hardware)

void setup() {
  // TODO: Initialize serial communication
  // Use Serial.begin(9600) - must match Python baud rate
  
  // TODO: Setup hardware pins
  // Set LED_PIN as OUTPUT
  
  // TODO: Setup your additional hardware
  // Examples:
  //   - Servo servos (include Servo.h and attach to pins)
  //   - LCD displays
  //   - Additional LEDs
  //   - Motors
  
  // TODO: Print ready message
  // Serial.println("Arduino Ready");
}

void loop() {
  // TODO: Read serial data character by character
  // while (Serial.available() > 0):
  //   1. Read one character with Serial.read()
  //   2. If character is '\n' (newline):
  //      a. Process the complete message
  //      b. Clear the buffer
  //   3. Otherwise, append character to inputBuffer
}

void processCommand(String cmd) {
  // TODO: Remove whitespace with cmd.trim()
  
  // TODO: Check if command is empty, return if so
  
  // TODO: Parse command type and call appropriate handler
  // Check if cmd.startsWith("PERSON:"):
  //   - Extract name with cmd.substring(7)
  //   - Call handlePersonRecognized(name)
  // Check if cmd.startsWith("COORD:"):
  //   - Find comma position with cmd.indexOf(',', 6)
  //   - Extract x coordinate: cmd.substring(6, commaIdx).toInt()
  //   - Extract y coordinate: cmd.substring(commaIdx + 1).toInt()
  //   - Call handleCoordinates(x, y)
  // Check if cmd == "UNKNOWN":
  //   - Call handleUnknown()
  
  // TODO: Send acknowledgment back to Jetson
  // Serial.println("ACK:" + cmd);
}

void handlePersonRecognized(String name) {
  // TODO: Implement hardware control based on recognized person
  // Examples:
  //   - Flash LED different patterns for different people
  //   - Display name on LCD screen
  //   - Play different sounds/melodies
  //   - Move servo to greet specific person
  //   - Turn on specific colored LED
  
  // Example structure:
  // if (name == "ben") {
  //     // Flash LED 3 times
  // }
  // else if (name == "james") {
  //     // Flash LED 2 times
  // }
  // else {
  //     // Default action
  // }
}

void handleCoordinates(int x, int y) {
  // TODO: Use coordinates to control hardware
  // Examples:
  //   - Map x coordinate to servo pan angle (0-180)
  //   - Map y coordinate to servo tilt angle (0-180)
  //   - Use to aim camera or turret at face
  
  // Example servo control:
  // int servoAngle = map(x, 0, 640, 0, 180);
  // myServo.write(servoAngle);
}

void handleUnknown() {
  // TODO: Implement unknown person response
  // Examples:
  //   - Turn on red warning LED
  //   - Display "Unknown" on screen
  //   - Play warning sound
  //   - Keep LED on for 1 second
}

