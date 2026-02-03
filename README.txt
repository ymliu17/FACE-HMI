═══════════════════════════════════════════════════════════════
Installation
═══════════════════════════════════════════════════════════════

pip install -r /path/to/requirement.txt

NOTE: Use 'pip3' for MacOS
NOTE: Some of the dependencies may not be listed, you have to manually install them based on the error info.

═══════════════════════════════════════════════════════════════
Work Online
═══════════════════════════════════════════════════════════════

Prepare:
* make sure the ''' WEBSERVER = "https://brainwellnessgamesmasterapi.azurewebsites.net" ''' in /your_path_to_xxx/FACE-Pilot/src/face2_api.py
** connect the ECG device to the laptop; make sure the ''' ECG_DEVICE_ADDRESS = "xxx" ''' in /your_path_to_xxx/FACE-Pilot/src/face2_api.py is correct (you can find the address in the back of the sensor).

Run:
1. open a new browser tab, start 'Master Game'-'Updated UI' in "https://brainwellnessgames.azurewebsites.net/", enter Session ID and submit session.
2. cd to FACE-Pilot dir, open a new terminal and type "python src/face2_api.py <SESSION_EXTERNAL_ID> <GROUP_ARM_ID>".
NOTE: '0' is for control group, '1' is for FACE group
3. waiting script running and click 'Start' in the browser.

═══════════════════════════════════════════════════════════════
Work Locally
═══════════════════════════════════════════════════════════════

Prepare (IMPORTANT):
* make sure the ''' WEBSERVER = "http://localhost:8080" ''' in /your_path_to_xxx/FACE-Pilot/src/face2_api.py
** connect the ECG device to the laptop; make sure the ''' ECG_DEVICE_ADDRESS = "xxx" ''' in /your_path_to_xxx/FACE-Pilot/src/face2_api.py is correct (you can find the address in the back of the sensor).


A. Start the local API using Docker Desktop UI:
    -----------------------
    START:
    1. Open Docker Desktop
    2. Go to "Containers" tab
    3. Find the container group
    4. Click the Play ▶ button

    STOP:
    1. Go to "Containers" tab
    2. Click the Stop ⏹ button (or trash icon to remove)

    VIEW LOGS:
    1. Click on a container name
    2. Click "Logs" tab
    3. Look for "Database restored successfully!" message

B. Start the FACE-API:
    -----------------------
    1. open a new browser tab, start 'Master' in "http://localhost:8082/", enter Session ID and submit session.
    (2a. Activate 'face' environment "conda activate face" if you use the FACE laptop.) 
    2. cd to FACE-Pilot dir, open a new terminal and type "python src/face2_api.py <SESSION_EXTERNAL_ID> <GROUP_ARM_ID>".
    NOTE: '0' is for control group, '1' is for FACE group
    3. waiting script running and click 'Start' in the browser.