import asyncio
import logging
from excavator import Excavator

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting the Excavator system")

    # Create an Excavator instance
    excavator = Excavator('configuration_files/excavator_config.yaml')

    # Print input mappings for debugging
    excavator.print_input_mappings()

    try:
        # Start the excavator system
        await excavator.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # Ensure the excavator system is properly stopped
        await excavator.stop()
        logger.info("Excavator system has been shut down")

if __name__ == "__main__":
    asyncio.run(main())