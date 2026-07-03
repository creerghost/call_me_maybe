import argparse
from .loader import Loader
from .catch import catch
from .llm import LLM

parser = argparse.ArgumentParser()
parser.add_argument("--functions_definition", help="Path to "
                                                   "functions definition "
                                                   "json file")
parser.add_argument("--input", help="Path to function calling json file")
parser.add_argument("--output", help="Path to output file")
parser.add_argument("--llm_path", help="Path to LLM")
parser.add_argument("--llm_name", help="Name of LLM model (name of class)")
args = parser.parse_args()

loader = catch(Loader, args.functions_definition, args.input)
llm = catch(LLM, args.llm_path, args.llm_name)
# print(f"Vocab size: {llm.get_vocab_size()}")
# print(f"Token for '{{': {llm.token2id.get('{', 'NOT FOUND')}")
# print(f"Token for 'true': {llm.token2id.get('true', 'NOT FOUND')}")
# print(f"Token ID 279 = '{llm.id2token.get(279, 'NOT FOUND')}'")
