
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
from transformers import AutoTokenizer
from .start_fastchat_api import start_fastchat_api
import dotenv
from .chatgpt import get_token_limit, _build_openai_client
import tiktoken

dotenv.load_dotenv()
HF_TOKEN=os.getenv("HF_TOKEN")

class LLMProxy(object):
    
    @staticmethod
    def regist_args(parser):
        parser.add_argument('--model_name', type=str, default='Llama-2-7b-chat-hf') # Llama-2-7b-chat-hf
        parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
        parser.add_argument("--conv_template", type=str, default="llama-2")
        parser.add_argument("--host", type=str, default="localhost")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--disable_auto_start", action="store_true")
        parser.add_argument('--retry', type=int, help="retry time", default=5)
        
    def __init__(self, args) -> None:
        self.args = args
        if "gpt-4" in args.model_name or "gpt-3.5" in args.model_name:
            # Load key for OpenAI API
            load_dotenv()
            openai.api_key = os.getenv("OPENAI_API_KEY")
            openai.organization = os.getenv("OPENAI_ORG")
            self.maximun_token = get_token_limit(self.model_name)
        else:
            # Use local API
            if not args.disable_auto_start:
                start_fastchat_api(args.model_name, args.model_path, args.conv_template, args.host, args.port)
            openai.api_key = "EMPTY"
            openai.base_url = f"http://{args.host}:{args.port}/v1"
        self.retry = args.retry
        self.model_name = args.model_name
        
    def prepare_for_inference(self):
        client = _build_openai_client(
            api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
        )
        self.client = client
        if "gpt-4" not in self.model_name and "gpt-3.5" not in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, token=HF_TOKEN,
            trust_remote_code=True, 
            use_fast=False)
            self.maximun_token = self.tokenizer.model_max_length
    
    def token_len(self, text):
        """Returns the number of tokens used by a list of messages."""
        if "gpt-4" in self.model_name or "gpt-3.5" in self.model_name:
            try:
                encoding = tiktoken.encoding_for_model(self.model_name)
                num_tokens = len(encoding.encode(text))
            except KeyError:
                raise KeyError(f"Warning: model {self.model_name} not found.")
        else:
            num_tokens = len(self.tokenizer.tokenize(text))
        return num_tokens
    
    def _calc_cost(self, prompt_tokens, completion_tokens):
        # Pricing is only available for OpenAI-hosted models; default to 0 otherwise.
        base_name = self.model_name.split("/")[-1]
        if "gpt-4" in base_name or "gpt-4" in self.model_name:
            return 0.03 * prompt_tokens / 1000.0 + 0.06 * completion_tokens / 1000.0
        if "gpt-3.5" in base_name or "gpt-3.5" in self.model_name:
            return 0.0005 * prompt_tokens / 1000.0 + 0.0015 * completion_tokens / 1000.0
        return 0.0

    def generate_sentence(self, llm_input, return_usage=False):
        query = [{"role": "user", "content": llm_input}]
        cur_retry = 0
        num_retry = self.retry
        # Chekc if the input is too long
        input_length = self.token_len(llm_input)
        if input_length > self.maximun_token:
            print(f"Input lengt {input_length} is too long. The maximum token is {self.maximun_token}.\n Right tuncate the input to {self.maximun_token} tokens.")
            llm_input = llm_input[:self.maximun_token]
        while cur_retry <= num_retry:
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = query,
                    timeout=60,
                    temperature=0.0
                    )
                elapsed = time.time() - start_time
                result = response.choices[0].message.content.strip() # type: ignore
                if not return_usage:
                    return result

                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
                completion_tokens = getattr(
                    usage, "completion_tokens", None) if usage else None
                total_tokens = getattr(usage, "total_tokens", None) if usage else None

                if prompt_tokens is None:
                    prompt_tokens = self.token_len(llm_input)
                if completion_tokens is None:
                    completion_tokens = self.token_len(result)
                if total_tokens is None:
                    total_tokens = prompt_tokens + completion_tokens

                usage_payload = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "elapsed": elapsed,
                    "cost": self._calc_cost(prompt_tokens, completion_tokens),
                    "calls": 1,
                }
                return result, usage_payload
            except Exception as e:
                print("Message: ", llm_input)
                print("Number of token: ", self.token_len(llm_input))
                print(e)
                time.sleep(30)
                cur_retry += 1
                continue
        return None
