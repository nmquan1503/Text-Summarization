from dotenv import load_dotenv
import os

load_dotenv()

VOCAB_PATH = os.getenv('VOCAB_PATH')
MODEL_CONFIG_PATH = os.getenv('MODEL_CONFIG_PATH')
MODEL_WEIGTHS_PATH = os.getenv('MODEL_WEIGHTS_PATH')
MAX_DOC_LENGTH = int(os.getenv('MAX_DOC_LENGTH'))
MAX_SENT_LENGTH = int(os.getenv('MAX_SENT_LENGTH'))
MAX_OUTPUT_LENGTH = int(os.getenv('MAX_OUTPUT_LENGTH'))
BEAM_SIZE = int(os.getenv('BEAM_SIZE'))