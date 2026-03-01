import tiktoken
from typing import NamedTuple

class Tokens(NamedTuple):
  encoded: str
  decoded: str

def tokens_from_string(string: str, model_name: str):
  encoding = tiktoken.encoding_for_model(model_name)
  encoded_tokens = encoding.encode(string)
  decoded_tokens = [encoding.decode_single_token_bytes(token) for token in encoded_tokens]
  return Tokens(encoded_tokens, decoded_tokens)

text_to_count = "The electroencephalographically hypercharacterization of institutionalization-related neurodevelopmental maladaptations demonstrates counterintuitive heteroscedasticity within psychoneuroimmunological frameworks. Furthermore, disproportionableness and deinstitutionalization-induced derecontextualization exacerbate mischaracterizations of antidisestablishmentarianism-like sociopolitical hyperfragmentation. Pseudopseudohypoparathyroidism, incomprehensibilities, and subdermatoglyphic microvascularization collectively illustrate the tokenizer’s probabilistic subword segmentation behavior."
model_used = "gpt-4o"

tokens = tokens_from_string(text_to_count, model_used)
num_tokens = len(tokens.encoded)


print(f'Total tokens: {num_tokens}')
print(f'Encoded tokens: {tokens.encoded}')
print(f'Decoded tokens: {tokens.decoded}')