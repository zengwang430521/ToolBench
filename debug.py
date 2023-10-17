from toolbench.inference.LLM.tool_llama_model import ToolLLaMA
ToolLLaMA(base_name_or_path='ToolBench/ToolLLaMA-7b')

import transformers

transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_2
transformers.models.llama2

from transformers import LlamaForCausalLM, LlamaTokenizer

print('finish')