import asyncio
import aiohttp
import random
import time
import hashlib
import json
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExcavatorTester:
    def __init__(self):
        self.server_url = "http://192.168.0.131:8000"  # Updated with port number
        self.num_values = 31
        self.client_id = "excavator"  # Changed to match expected client ID
        self.http_session = None

    @staticmethod
    def calculate_checksum(values, timestamp):
        data = json.dumps(values) + str(timestamp)
        return hashlib.md5(data.encode()).hexdigest()

    async def setup(self):
        logger.info("Setting up Excavator Tester...")
        self.http_session = aiohttp.ClientSession()

    async def run(self):
        logger.info("Starting Excavator Tester...")
        try:
            send_task = asyncio.create_task(self.send_loop())
            receive_task = asyncio.create_task(self.receive_loop())
            await asyncio.gather(send_task, receive_task)
        except asyncio.CancelledError:
            logger.info("Asyncio task cancelled. Shutting down...")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
        finally:
            await self.stop()

    async def send_loop(self):
        while True:
            await self.send_random_values()
            await asyncio.sleep(1)  # Send every second

    async def receive_loop(self):
        while True:
            await self.receive_values()
            await asyncio.sleep(1)  # Receive every second

    async def send_random_values(self):
        values = [random.uniform(-1, 1) for _ in range(self.num_values)]
        timestamp = time.time()
        checksum = self.calculate_checksum(values, timestamp)

        payload = {
            "values": values,
            "timestamp": timestamp,
            "checksum": checksum
        }

        try:
            async with self.http_session.post(f"{self.server_url}/send/{self.client_id}", json=payload) as response:
                if response.status == 200:
                    logger.info(f"Sent {self.num_values} random values for {self.client_id}")
                    response_data = await response.json()
                    logger.info(f"Server response: {response_data['status']}")
                else:
                    logger.error(f"Failed to send values. Status code: {response.status}")
                    response_text = await response.text()
                    logger.error(f"Server message: {response_text}")
        except Exception as e:
            logger.error(f"Error sending values: {e}")

    async def receive_values(self):
        try:
            async with self.http_session.get(f"{self.server_url}/receive/{self.client_id}") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Received values: {data['values']}")
                    logger.info(f"Timestamp: {data['timestamp']}")
                    logger.info(f"Checksum: {data['checksum']}")

                    calculated_checksum = self.calculate_checksum(data['values'], data['timestamp'])
                    if calculated_checksum == data['checksum']:
                        logger.info("Checksum verified")
                    else:
                        logger.warning("Checksum mismatch!")
                elif response.status == 204:
                    logger.info("No new data available")
                else:
                    logger.error(f"Failed to receive values. Status code: {response.status}")
                    response_text = await response.text()
                    logger.error(f"Server message: {response_text}")
        except Exception as e:
            logger.error(f"Error receiving values: {e}")

    async def stop(self):
        logger.info("Stopping Excavator Tester...")
        if self.http_session:
            await self.http_session.close()

async def main():
    tester = ExcavatorTester()
    try:
        await tester.setup()
        await tester.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    finally:
        await tester.stop()

if __name__ == "__main__":
    asyncio.run(main())