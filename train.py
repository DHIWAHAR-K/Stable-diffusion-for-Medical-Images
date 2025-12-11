import logging
import sys
import sys
from config import load_config, get_args
from engine import Trainer

def main():
    # Logging Setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    args = get_args()
    logger.info(f"Loading config from {args.config}")
    
    try:
        config = load_config(args.config)
        trainer = Trainer(config)
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

