import requests
import time
import cv2
import sys
import os
from pipeline import get_block_fatigue, make_decision

requestTime = 0.5
sessionExternalId = sys.argv[1]  # "ct_test" # TODO: Read this from the input

webserver = "https://brainwellnessgamesmasterapi.azurewebsites.net"

print(sessionExternalId)

session_getIdentifier_api_url = "{}/session/GetByExternalIdentifier?externalIdentifier={}".format(webserver, sessionExternalId)

# GET
identifierResponse = requests.get(session_getIdentifier_api_url)

print(identifierResponse.status_code)  # It's good practice to check if this is a 200 or 201
print(identifierResponse.json())

sessionId = identifierResponse.json()["identifier"]

device_updateStatus_api_url = "{}/device/UpdateStatus".format(webserver)
device_saveResult_api_url = "{}/device/SaveExtendedResult".format(webserver)

print(sessionId)

block_getLatestForSession_api_url = "{}/block/GetLatestForSession?sessionIdentifier={}".format(webserver, sessionId)
block_completeCreateNext_api_url = "{}/block/CompleteAndCreateNext".format(webserver)

# These are the game identifiers
'''
game_FK are as follows:
1 - Sound Sweeps
2 - Target Tracker
3 - Mixed Signals
4 - Delayed Task Switching (or Task 4)

targetTrackerIdentifier = "52524eab-69f6-43ad-b680-4ac1ef21b517"
mixedSignalsIdentifier = "f1365d89-cc5a-415e-a9f7-f5cfe697adff"
soundSweepIdentifier = "4c9421ae-e979-44f1-8c88-01bf5c195c92"
task4Identifier = "05561BA9-5DB1-4F15-9C99-8090EACFCB42"
'''
games = {
    1: "4c9421ae-e979-44f1-8c88-01bf5c195c92",
    2: "52524eab-69f6-43ad-b680-4ac1ef21b517",
    3: "f1365d89-cc5a-415e-a9f7-f5cfe697adff",
    4: "05561BA9-5DB1-4F15-9C99-8090EACFCB42"
}

# consecutive low accuracy marker
consec = False

while True:
    print("Starting new block")

    # GET
    latestBlockResponse = requests.get(block_getLatestForSession_api_url)

    print(latestBlockResponse.json())
    print(latestBlockResponse.status_code)  # It's good practice to check if this is a 200 or 201

    blockId = latestBlockResponse.json()["identifier"]
    gameId = latestBlockResponse.json()["game_FK"]
    block_idx = latestBlockResponse.json()["block_ID"]
    # currGameIdentifier = games[gameId]
    currLevel = latestBlockResponse.json()["startLevel"]
    # print("Current game identifier:", currGameIdentifier)
    print("Current level:", currLevel)

    game_checkStatus_api_url = "{}/game/checkstatus?sessionId={}".format(webserver, blockId)

    # GET
    gameReadyResponse = requests.get(game_checkStatus_api_url)

    print(gameReadyResponse.json())
    print(gameReadyResponse.status_code)  # It's good practice to check if this is a 200 or 201

    isGameReady = gameReadyResponse.json()["isGameReady"]

    # Loop until the game is ready
    while not isGameReady:
        gameReadyResponse = requests.get(game_checkStatus_api_url)

        print(gameReadyResponse.json())
        print(gameReadyResponse.status_code)  # It's good practice to check if this is a 200 or 201
        isGameReady = gameReadyResponse.json()["isGameReady"]
        time.sleep(requestTime)

    print("Game is ready:", isGameReady)

    # Start Device acquisition (video) and notify web API that the device is ready after device is ready
    video_path = os.path.join("Pilot", f"ses-{sessionExternalId}", f"block-{block_idx}.mp4")  # Example path
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        break

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (1920, 1080))

    # Notify the API that the device is Ready
    deviceReady = {"isReady": True, "sessionIdentifier": "{}".format(blockId)}
    deviceReadyResponse = requests.put(device_updateStatus_api_url, json=deviceReady)
    
    print(deviceReadyResponse.status_code)  # It's good practice to check if this is a 200 or 201

    # Loop until the user is playing the game
    while isGameReady:
        # start the video recording
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read video.")
            break

        gameReadyResponse = requests.get(game_checkStatus_api_url)
        isGameReady = gameReadyResponse.json()["isGameReady"]
        print("User is still playing...", gameReadyResponse.json())

        out.write(frame)

    # Stop Device acquisition (video?) and notify web API that the device is not ready after device is ready
    cap.release()
    out.release()
    print("Saving video to:", video_path)

    # Change Device status to notReady
    deviceReady = {"isReady": False, "sessionIdentifier": "{}".format(blockId)}
    deviceReadyResponse = requests.put(device_updateStatus_api_url, json=deviceReady)

    # Get the results of the game
    game_getResultForBlock_api_url = "{}/game/GetResult?blockIdentifier={}".format(webserver, blockId)
    gameResultResponse = requests.get(game_getResultForBlock_api_url)

    # print(gameResultResponse.json())
    # resultsOfLastBlock = gameResultResponse.json()["results"]
    accuracyOfLastBlock = gameResultResponse.json()["accuracy"]
    print("Accuracy of last block:", accuracyOfLastBlock)
    print(gameResultResponse.status_code)  # It's good practice to check if this is a 200 or 201

    # Fatigue detection and decision making
    face_outdir = os.path.join("Pilot", f"ses-{sessionExternalId}", f"block-{block_idx}")  # Example path
    os.makedirs(face_outdir, exist_ok=True)
    model_dir = "pre_model/model.pth"  # Example model path

    fatigueOfLastBlock = get_block_fatigue(video_path, face_outdir, model_dir)
    if fatigueOfLastBlock is None:
        print("No fatigue score could be inferred.")
        continue
    print("Fatigue of last block:", fatigueOfLastBlock)

    # Make decision based on fatigue and accuracy
    nextGame, nextLevel, consec = make_decision(block_idx, fatigueOfLastBlock, accuracyOfLastBlock, gameId, currLevel, consec)
    nextGameIdentifier = games[nextGame]

    print(f"New game: {nextGameIdentifier}, New level: {nextLevel}")

    faceResult = {"NextGameIdentifier": "{}".format(nextGameIdentifier), "NextLevel": nextLevel, "BlockIdentifier": "{}".format(blockId)}
    deviceResultResponse = requests.put(device_saveResult_api_url, json=faceResult)

    print(deviceResultResponse.json())
    print(deviceResultResponse.status_code)  # It's good practice to check if this is a 200 or 201

    while not isGameReady:
        gameReadyResponse = requests.get(game_checkStatus_api_url)

        print("Game is not ready. It should be creating a new block for the current session...", gameReadyResponse.json())
        isGameReady = gameReadyResponse.json()["isGameReady"]
        time.sleep(requestTime)

print("And we are done with this session!")
