"""
Helper script to find Arduino serial port on Jetson/Linux.
"""
import serial.tools.list_ports

def find_arduino_port():
    """List all available serial ports"""
    print("Available serial ports:")
    print("="*50)
    
    # TODO: Get list of all serial ports
    # Use serial.tools.list_ports.comports()
    
    # TODO: Check if any ports found
    # If empty, print "No serial ports found!" and return
    
    # TODO: Loop through each port and print information
    # For each port, print:
    #   - Port device name (port.device)
    #   - Description (port.description)
    #   - Manufacturer (port.manufacturer)
    #   - Check if 'Arduino' or 'USB' in description
    #     If yes, print "✅ Likely Arduino!"
    
    # TODO: Print helpful summary
    # Print common port names:
    #   - /dev/ttyUSB0 (USB-to-Serial adapter)
    #   - /dev/ttyACM0 (Arduino Uno/Nano)
    # Remind user to update configs/config.yaml

if __name__ == '__main__':
    find_arduino_port()

