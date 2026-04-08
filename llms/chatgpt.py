import time
import os
import random
import threading
from openai import OpenAI
import openai
from .base_language_model import BaseLanguageModel
import dotenv
import tiktoken
import glob

from utils_windows_long_path import maybe_windows_long_path
dotenv.load_dotenv()

os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'

OPENAI_MODEL = ['gpt-4', 'gpt-3.5-turbo']


def get_token_limit(model='gpt-4'):
    """Returns the token limitation of provided model"""
    if model in ['gpt-4', 'gpt-4-0613']:
        num_tokens_limit = 8192
    elif model in ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo']:
        num_tokens_limit = 16384
    elif model in ['gpt-3.5-turbo-0613', 'text-davinci-003', 'text-davinci-002']:
        num_tokens_limit = 4096
    elif model in ['gpt-5-nano', 'gpt-5.2']:
        num_tokens_limit = 400000
    else:
        raise NotImplementedError(f"""get_token_limit() is not implemented for model {model}.""")
    return num_tokens_limit


PROMPT = """{instruction}

{input}"""


def _build_openai_client(**kwargs):
    try:
        return OpenAI(**kwargs)
    except TypeError as e:
        # openai==1.9.0 passes `proxies=` to httpx.Client, but httpx>=0.28 renamed it to `proxy=`,
        # causing: "Client.__init__() got an unexpected keyword argument 'proxies'".
        if "unexpected keyword argument 'proxies'" not in str(e):
            raise
        import httpx  # type: ignore

        return OpenAI(**kwargs, http_client=httpx.Client())


class _TimeBasedGlobalLimiter:
    def __init__(self, rps: float):
        self._lock = threading.Lock()
        self._rps = float(rps)
        self._interval = 1.0 / self._rps if self._rps > 0 else 0.0
        self._next_allowed_ts = 0.0

    def acquire(self):
        if self._interval <= 0:
            return

        sleep_for = 0.0
        with self._lock:
            now = time.monotonic()
            if self._next_allowed_ts <= 0.0:
                self._next_allowed_ts = now

            if now < self._next_allowed_ts:
                sleep_for = self._next_allowed_ts - now
                reserved_ts = self._next_allowed_ts
            else:
                reserved_ts = now

            self._next_allowed_ts = reserved_ts + self._interval

        if sleep_for > 0:
            time.sleep(sleep_for)


_GLOBAL_LIMITER = None
_GLOBAL_LIMITER_INIT_LOCK = threading.Lock()


def _get_global_limiter(rps: float):
    global _GLOBAL_LIMITER
    if _GLOBAL_LIMITER is None:
        with _GLOBAL_LIMITER_INIT_LOCK:
            if _GLOBAL_LIMITER is None:
                _GLOBAL_LIMITER = _TimeBasedGlobalLimiter(rps)
    return _GLOBAL_LIMITER


def _sleep_with_backoff(retry: int, base: float = 1.0, cap: float = 30.0):
    sleep_time = min(cap, base * (2 ** retry)) * random.uniform(0.7, 1.3)
    time.sleep(sleep_time)


class ChatGPT(BaseLanguageModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--retry', type=int, help="retry time", default=5)
        parser.add_argument('--model_path', type=str, default='None')
        parser.add_argument('--or_rps', type=float, default=2.5,
                            help="OpenRouter global requests/sec (process-wide)")
        parser.add_argument('--or_timeout', type=int, default=120,
                            help="OpenRouter request timeout (seconds)")

    def __init__(self, args):
        super().__init__(args)
        self.retry = args.retry
        self.model_name = args.model_name
        self.maximum_token = get_token_limit(self.model_name)
        self._timeout = getattr(args, "or_timeout", 120)
        self._limiter = _get_global_limiter(getattr(args, "or_rps", 2.5))

    def token_len(self, text):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # Fallback for new/unknown model names (e.g., OpenRouter aliases like "gpt-5.2").
            # Token counts are only used for rough length checks/cost estimates, so an approximate
            # encoder is fine here.
            encoding = None
            for enc_name in ("o200k_base", "cl100k_base"):
                try:
                    encoding = tiktoken.get_encoding(enc_name)
                    break
                except Exception:
                    continue
            if encoding is None:
                # Last resort: rough heuristic (~4 chars/token).
                return max(1, len(text) // 4)

        return len(encoding.encode(text))

    def prepare_for_inference(self, model_kwargs={}):
        client = _build_openai_client(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ['OPENROUTER_API_KEY'],  # this is also the default, it can be omitted
        )
        self.client = client

    def prepare_model_prompt(self, query):
        '''
        Add model-specific prompt to the input
        '''
        return query

    def generate_sentence(self, llm_input, return_usage=False):
        query = [{"role": "user", "content": llm_input}]
        cur_retry = 0
        num_retry = self.retry
        # Check if the input is too long
        input_length = self.token_len(llm_input)
        if input_length > self.maximum_token:
            print(
                f"Input length {input_length} is too long. The maximum token is {self.maximum_token}.\n Right truncate the input to {self.maximum_token} tokens.")
            llm_input = llm_input[:self.maximum_token]
            query = [{"role": "user", "content": llm_input}]
        while cur_retry <= num_retry:
            try:
                self._limiter.acquire()
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model="openai/" + self.model_name,
                    messages=query,
                    timeout=self._timeout,
                    temperature=0.0
                )
                elapsed = time.time() - start_time
                result = response.choices[0].message.content.strip()  # type: ignore
                if not return_usage:
                    return result

                # print(response)
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
                completion_tokens = getattr(
                    usage, "completion_tokens", None) if usage else None
                total_tokens = getattr(usage, "total_tokens", None) if usage else None
                cost = getattr(usage, "cost", None) if usage else None

                if prompt_tokens is None:
                    prompt_tokens = self.token_len(llm_input)
                if completion_tokens is None:
                    completion_tokens = self.token_len(result)
                if total_tokens is None:
                    total_tokens = prompt_tokens + completion_tokens
                if cost is None and usage and getattr(usage, "cost_details", None):
                    details = getattr(usage, "cost_details")
                    if isinstance(details, dict):
                        numeric_vals = [v for v in details.values() if isinstance(v, (int, float))]
                        if numeric_vals:
                            cost = sum(numeric_vals)
                    else:
                        try:
                            fields = [
                                getattr(details, "upstream_inference_cost", None),
                                getattr(details, "upstream_inference_prompt_cost", None),
                                getattr(details, "upstream_inference_completions_cost", None),
                            ]
                            numeric_vals = [v for v in fields if isinstance(v, (int, float))]
                            if numeric_vals:
                                cost = sum(numeric_vals)
                        except Exception:
                            cost = None
                if cost is None:
                    cost = 0.0

                usage_payload = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "elapsed": elapsed,
                    "cost": cost,
                    "calls": 1,
                }
                return result, usage_payload
            except openai.APITimeoutError as e:
                print("Request Time out. Retrying...")
                _sleep_with_backoff(cur_retry)
                cur_retry += 1
            except openai.RateLimitError as e:
                print("Rate limit exceeded. Retrying...")
                _sleep_with_backoff(cur_retry)
                cur_retry += 1
            except openai.APIConnectionError as e:
                # 打印异常的详细信息
                print("Failed to connect to OpenAI API.")
                print("Error message:", e.args[0] if e.args else "No details available.")
                if hasattr(e, 'response') and e.response:
                    print("HTTP response status:", e.response.status_code)
                    print("HTTP response body:", e.response.text)
                else:
                    print("No HTTP response received.")
                print("Retrying...")
                _sleep_with_backoff(cur_retry)
                cur_retry += 1
            except Exception as e:
                print(e)
                print("Number of token: ", self.token_len(llm_input))
                _sleep_with_backoff(cur_retry)
                cur_retry += 1
        print(f"Maximum retries reached. Unable to generate sentence")
        return None

    def gen_rule_statistic(self, input_dir, output_file_path):
        sum = 0
        with open(output_file_path, 'w') as fout:
            for input_filepath in glob.glob(maybe_windows_long_path(os.path.join(input_dir, "*.txt"))):
                file_name = input_filepath.split("/")[-1]
                if file_name.startswith('fail'):
                    continue
                else:
                    with open(input_filepath, 'r') as fin:
                        rules = fin.readlines()
                        for rule in  rules:
                            if 'Rule_head' in rule:
                                continue
                            elif 'Sample' in rule:
                                continue
                            fout.write(rule)
                            sum+=1

            fout.write(f"LL {sum}\n")
