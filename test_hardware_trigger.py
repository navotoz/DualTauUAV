# See README

import tkinter as tk
import RPi.GPIO as GPIO
import threading
import time
from collections import deque

from devices.Tau2Grabber import Tau2


# Define the length of the deque for measuring the true sampling rate
N = 10

# Define the global variables for the trigger rate, the camera deques, and the lock object
trigger_freq = 30  # Hz
duty_cycle = 0.5
period = 1 / trigger_freq
on_time = period * duty_cycle  # seconds
off_time = period - on_time  # seconds
cam1_deque = deque(maxlen=N)
cam2_deque = deque(maxlen=N)


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
def camera_loop(cam, cam_deque: deque):
    while True:
        # Try to get a frame from the camera
        # If the frame is not None, append the current time to the deque
        if cam.frame() is not None:
            cam_deque.append(time.time_ns())


# Define a function to calculate the true sampling rate from a deque of timestamps
def get_rate(cam_deque: deque):
    # If the deque is not full, return None
    if len(cam_deque) < min(2, N):
        return None
    # Otherwise, calculate the rate as N / (last - first)
    else:
        return 1e9 * N / (cam_deque[-1] - cam_deque[0])


# Define a function to update the GUI with the true sampling rates
def update_gui():
    global cam1_deque, cam2_deque
    # Get the rates from the deques
    cam1_rate = get_rate(cam1_deque)
    cam2_rate = get_rate(cam2_deque)
    # Update the labels with the rates or N/A if None
    if cam1_rate is None:
        cam1_label.config(text="Camera 1: N/A")
    else:
        cam1_label.config(text=f"Camera 1: {cam1_rate:.2f} Hz")
    if cam2_rate is None:
        cam2_label.config(text="Camera 2: N/A")
    else:
        cam2_label.config(text=f"Camera 2: {cam2_rate:.2f} Hz")
    # Schedule the next update after 100 ms
    root.after(100, update_gui)


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


# Init cameras
camera_1 = Tau2(name='1')
camera_2 = Tau2(name='2')

# Create the GUI using tkinter
root = tk.Tk()
root.title("Camera Trigger Test")

# Create a slider to control the trigger rate
slider_freq = tk.Scale(root, from_=1, to=40, orient=tk.HORIZONTAL, command=set_trigger_rate)
slider_freq.set(trigger_freq)
slider_freq.pack()

# Create a slider to control the duty cycle
slider_duty = tk.Scale(root, from_=0.01, to=0.99, resolution=0.01, orient=tk.HORIZONTAL, command=set_duty_cycle)
slider_duty.set(duty_cycle)
slider_duty.pack()

# Create labels to display the true sampling rates
cam1_label = tk.Label(root, text="Camera 1: N/A")
cam1_label.pack()
cam2_label = tk.Label(root, text="Camera 2: N/A")
cam2_label.pack()

# Initialize the GPIO pin for the trigger signal
PIN_TRIGGER = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_TRIGGER, GPIO.OUT)

# Create and start the threads for the trigger loop and the camera loops
trigger_thread = threading.Thread(target=trigger_loop, daemon=True)
trigger_thread.start()
cam1_thread = threading.Thread(target=camera_loop, args=(camera_1, cam1_deque), daemon=True)
cam1_thread.start()
cam2_thread = threading.Thread(target=camera_loop, args=(camera_2, cam2_deque), daemon=True)
cam2_thread.start()

# Start the GUI update loop
update_gui()

# Start the main loop of the GUI
root.mainloop()

# Clean up the GPIO pin
GPIO.cleanup()
