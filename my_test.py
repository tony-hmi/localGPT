from llama_cpp import Llama

prompt = """[INST] <<SYS>>
Name the planets in the solar system? 
<</SYS>>
[/INST] 
"""


for ngl in range(35+1):
    print('ngl %d' % ngl)
    llm = Llama(model_path="models/models--TheBloke--zephyr-7B-alpha-GGUF/snapshots/17246e1732130e712d6d53b46c2f5179d47e84e9/zephyr-7b-alpha.Q5_K_M.gguf", n_gpu_layers=ngl, n_ctx=4096, n_batch=512, verbose=True)
    output = llm(prompt, max_tokens=350, echo=True)
    print(output['choices'][0]['text'].split('[/INST]')[-1])