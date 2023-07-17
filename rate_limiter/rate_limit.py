from RPi import GPIO
from threading import Thread
from time import sleep


# The sampling rate of the cameras
FREQUENCY_CAMERAS = 30  # Hz
DUTY_CYCLE = 0.5
period = 1 / FREQUENCY_CAMERAS
low_time = period * DUTY_CYCLE  # seconds
high_time = period - low_time  # seconds

# Initialize the GPIO pin for the trigger signal
PIN_TRIGGER = 17


def th_rpi_trigger_for_cam():
    # Define a function to toggle the trigger signal at the given rate
    while True:
        # Set the trigger pin to high
        GPIO.output(PIN_TRIGGER, GPIO.HIGH)
        # Wait for half the period
        sleep(high_time)
        # Set the trigger pin to low
        GPIO.output(PIN_TRIGGER, GPIO.LOW)
        # Wait for the other half of the period
        sleep(low_time)


if __name__ == '__main__':
    GPIO.setmode(GPIO.BCM)
    GPIO.cleanup()
    GPIO.setup(PIN_TRIGGER, GPIO.OUT)

    # Hardware rate setter
    th_hardware_trigger = Thread(target=th_rpi_trigger_for_cam, daemon=False, name='hardware_trigger')
    th_hardware_trigger.start()
    th_hardware_trigger.join()
