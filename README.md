# Ariel Dataset Collector

## Installation

1. Install Linux Ubuntu (tested using Ubuntu 22.04)
2. Install Python 3.10
3. Clone or Download the project.
4. Open the terminal (`Ctrl+D`)
5. Dependencies to install:
   1. `sudo apt-get install -y python3-pip`
   2. `sudo apt-get install python-tk python3-tk -y`
6. Add USB buffer to the system (See [link](https://www.flir.com/support-center/iis/machine-vision/application-note/understanding-usbfs-on-linux/))
   - ```sudo touch /etc/rc.local```
   - ```sudo chmod 744 /etc/rc.local```
   - Add:
   ```
   #!/bin/sh -e
   echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb
   exit 0
   ```
   - If file already exists just add the echo 1000... line before the exit 0 line.
   - Restart the computer.
   - Run ```cat /sys/module/usbcore/parameters/usbfs_memory_mb``` to check if the buffer was added.
7. Write `sudo su` and enter the computer's root password.
8. Write:
   `echo "SUBSYSTEM==\"usb\", ACTION==\"add\", ATTRS{idVendor}==\"0403\", ATTRS{idProduct}==\"6010\", MODE=\"0666\"">/etc/udev/rules.d/99-taucamera.rules`
9. Write:
   `sudo usermod -a -G dialout $USER`
10. If 1-5 doesn't work, check which USB the device is connected to via `lsusb` and write in terminal:
    `sudo chmod 666 #` Where # is the correct device address.
11. **Reboot**
12. Activate the venv by `source path/to/venv/bin/activate`
13. Install the requirements file using:
    `pip install -r requirements.txt`
14. Make sure to install on pip3 also.

# Aerial collector with RPi

- Clone the repository into a RPi.
- Perform all the installation steps above.
- Install Docker. Docker-compose is supposed to be installed automatically.
- Set docker service to start on boot:

```
sudo systemctl enable docker
```

- Run the docker-compose file:

```
docker compose up -d
```

- The `restart` option on the `docker-compose.yaml` is set to `always`, so the docker should run automatically on boot.

# Undervoltage in the RPi
- In terminal:
```vcgencmd get_throttled```
Bit	Hex Value	Meaning
0	1	Under-voltage detected
1	2	ARM frequency has been caped
2	4	Currently throttled
3	8	Soft temperature limit is active
16	1000	Under-voltage has occurred
17	2000	ARM frequency capping has occurred
18	4000	Throttling has occurred
19	8000	Soft temperature limit has occurred

https://pimylifeup.com/raspberry-pi-low-voltage-warning/