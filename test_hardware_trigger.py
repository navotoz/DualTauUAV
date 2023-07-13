# See README
import RPi.GPIO as GPIO
import threading
import time
from collections import deque

from devices.Tau2Grabber import Tau2


# Define the length of the deque for measuring the true sampling rate
N = 50

# Define the global variables for the trigger rate, the camera deques, and the lock object
trigger_freq = 30  # Hz
duty_cycle = 0.1
period = 1 / trigger_freq
on_time = period * duty_cycle  # seconds
off_time = period - on_time  # seconds
cam1_deque = deque(maxlen=N)
cam2_deque = deque(maxlen=N)
time_start = time.time_ns()
count_sync = 0


# Define a function to toggle the trigger signal at the given rate
def trigger_loop():
    global trigger_freq
    while True:
        # Set the trigger pin to high
        GPIO.output(PIN_TRIGGER, GPIO.HIGH)
        # Wait for half the period
        time.sleep(on_time)
        # Set the trigger pin to low
        GPIO.output(PIN_TRIGGER, GPIO.LOW)
        # Wait for the other half of the period
        time.sleep(off_time)


# Define a function to sample frames from a camera and append the timestamps to a deque
def camera_loop(cam: Tau2, cam_deque: deque):
    global time_start
    time_start = time.time_ns()
    while True:
        # Try to get a frame from the camera
        # If the frame is not None, append the current time to the deque
        if cam.grab() is not None:
            cam_deque.append(time.time_ns())


# Define a function to calculate the true sampling rate from a deque of timestamps
def get_rate(cam_deque: deque):
    # If the deque is not full, return None
    if len(cam_deque) < min(2, N):
        return 0
    # Otherwise, calculate the rate as N / (last - first)
    else:
        return 1e9 * N / (cam_deque[-1] - cam_deque[0])


# Define a function to set the trigger rate from the slider value
def set_trigger_rate(value):
    global trigger_freq, period, on_time, off_time, duty_cycle
    # Convert the value to an integer and assign it to the trigger rate
    trigger_freq = int(value)
    period = 1 / trigger_freq
    on_time = period * duty_cycle  # seconds
    off_time = period - on_time  # seconds
    cam1_deque.clear()
    cam2_deque.clear()


def set_duty_cycle(value):
    global duty_cycle, on_time, off_time
    duty_cycle = float(value)
    on_time = period * duty_cycle  # seconds
    off_time = period - on_time  # seconds
    cam1_deque.clear()
    cam2_deque.clear()


# Function to be called when a sync toggle is detected
def sync_toggle(channel):
    global previous_state, count_sync
    current_state = GPIO.input(channel)
    if current_state != previous_state:
        count_sync += 1
        previous_state = current_state


# Init cameras
class Dummy:
    def __init__(self, name) -> None:
        pass

    def frame(self):
        return ''



camera_1 = Tau2(name='1')
camera_2 = Tau2(name='2')

# Initialize the GPIO pin for the trigger signal
PIN_TRIGGER = 17
PIN_SYNC = 27
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_TRIGGER, GPIO.OUT)
GPIO.setup(PIN_SYNC, GPIO.IN, pull_up_down=GPIO.PUD_UP)
previous_state = GPIO.input(PIN_SYNC)

# Add event detection for falling and rising edges on the sync pin
GPIO.add_event_detect(PIN_SYNC, GPIO.BOTH, callback=sync_toggle)

# Get the trigger rate from the user
trigger_rate = float(input("Enter the trigger rate: "))
set_trigger_rate(trigger_rate)

# Create and start the threads for the trigger loop and the camera loops
trigger_thread = threading.Thread(target=trigger_loop, daemon=True)
trigger_thread.start()
cam1_thread = threading.Thread(target=camera_loop, args=(camera_1, cam1_deque), daemon=True)
cam1_thread.start()
cam2_thread = threading.Thread(target=camera_loop, args=(camera_2, cam2_deque), daemon=True)
cam2_thread.start()


# Update the progress bar based on the trigger rate
try:
    while True:
        print(f"Cam 1: {get_rate(cam_deque=cam1_deque):.1f}Hz, Cam 2: {get_rate(cam_deque=cam2_deque):.1f}Hz, "
              f"Sync {1e9 * count_sync / (time.time_ns() - time_start):.1f} | "
              f"Freq: {trigger_freq:.0f}, Duty cycle: {duty_cycle:.1f}")
        time.sleep(1)
except Exception as e:
    print(str(e))
finally:
    # Clean up the GPIO pin
    GPIO.cleanup()
