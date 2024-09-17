import requests
import random
import time
import hashlib
import json

# Configuration
SERVER_URL = "http://localhost:8000"  # Replace with the actual server IP when testing
NUM_VALUES = 20  # Number of values for the excavator
CLIENT_ID = "motion_platform"


def calculate_checksum(values, timestamp):
    data = json.dumps(values) + str(timestamp)
    return hashlib.md5(data.encode()).hexdigest()


def send_random_values():
    # Generate random values between -1 and 1
    values = [random.uniform(-1, 1) for _ in range(NUM_VALUES)]
    timestamp = time.time()
    checksum = calculate_checksum(values, timestamp)

    # Send the values to the server
    try:
        payload = {
            "values": values,
            "timestamp": timestamp,
            "checksum": checksum
        }
        response = requests.post(f"{SERVER_URL}/send/{CLIENT_ID}", json=payload)

        if response.status_code == 200:
            print(f"Sent {NUM_VALUES} random values for {CLIENT_ID}")
            response_data = response.json()
            print(f"Server response: {response_data['status']}")
        else:
            print(f"Failed to send values. Status code: {response.status_code}")
            response_data = response.json()
            print(f"Server message: {response_data['status']}")
    except requests.RequestException as e:
        print(f"Error sending values: {e}")


def receive_values():
    # Receive values from the server
    try:
        response = requests.get(f"{SERVER_URL}/receive/{CLIENT_ID}")

        if response.status_code == 200:
            data = response.json()
            print(f"Received values: {data['values']}")
            print(f"Timestamp: {data['timestamp']}")
            print(f"Checksum: {data['checksum']}")

            # Verify checksum
            calculated_checksum = calculate_checksum(data['values'], data['timestamp'])
            if calculated_checksum == data['checksum']:
                print("Checksum verified")
            else:
                print("Checksum mismatch!")
        else:
            print(f"Failed to receive values. Status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error receiving values: {e}")


def main():
    print("Starting Excavator Test Script")
    print(f"Server URL: {SERVER_URL}")
    print(f"Number of values: {NUM_VALUES}")

    while True:
        send_random_values()
        receive_values()

        # Uncomment the following lines if you want to add a delay between iterations
        # print("\nWaiting for 5 seconds before next round...\n")
        # time.sleep(5)


if __name__ == "__main__":
    main()