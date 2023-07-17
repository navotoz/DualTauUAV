import logging
from RPi import GPIO
from time import sleep


# The sampling rate of the cameras
FREQUENCY_CAMERAS = 30  # Hz
DUTY_CYCLE = 0.5
period = 1 / FREQUENCY_CAMERAS
low_time = period * DUTY_CYCLE  # seconds
high_time = period - low_time  # seconds

# Initialize the GPIO pin for the trigger signal
PIN_TRIGGER = 17


if __name__ == '__main__':
    logger = logging.getLogger('rateLimiter')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Create a stdout handler
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    # Add the stdout handler to the logger
    logger.addHandler(stdout_handler)
    
    GPIO.setmode(GPIO.BCM)
    try:
        GPIO.cleanup()
        GPIO.setup(PIN_TRIGGER, GPIO.OUT)
        logger.info('Setup complete, starting...')

        # Hardware rate setter
        while True:
            # Set the trigger pin to high
            GPIO.output(PIN_TRIGGER, GPIO.HIGH)
            # Wait for half the period
            sleep(high_time)
            # Set the trigger pin to low
            GPIO.output(PIN_TRIGGER, GPIO.LOW)
            # Wait for the other half of the period
            sleep(low_time)
    except Exception as e:
        print(str(e))
    finally:
        GPIO.cleanup()
