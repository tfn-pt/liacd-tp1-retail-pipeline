import logging
import sys

def get_logger(name: str = "TP1") -> logging.Logger:
    """Configura e retorna um logger estruturado SOTA para o terminal."""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        # Formato limpo e alinhado para facilitar o trace de 250k eventos
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-7s | %(module)-10s | %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_logger()
