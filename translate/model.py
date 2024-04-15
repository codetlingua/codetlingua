import json
import os
from abc import ABC, abstractmethod
from typing import List
from warnings import warn
import glob

cache_dir = os.getcwd() + "/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir

import anthropic
import anthropic_request
import google.generativeai as genai

import openai
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from openai_request import make_auto_request
from vllm import LLM, SamplingParams

EOS = ["<|endoftext|>", "<|endofmask|>", "</s>"]

# function to build prompt accroding to various model instruction formats
def compose_prompt(prompt_type: str, source_lang: str, target_lang: str, code: str) -> str:

    prompt = f"{source_lang}:\n{code}\n\nTranslate the above {source_lang} code to {target_lang} and end with comment \"<END-OF-CODE>\".\n\n{target_lang}:\n"    

    if prompt_type == 'gpt' or prompt_type == 'gemini':
        prompt = f'You are a code translation expert. Translate the {source_lang} code below to {target_lang}\n\n{source_lang}\n{code}\n\n{target_lang}\n'

    if prompt_type == 'claude':
        prompt = f'\n```Translate the {source_lang} code below to {target_lang}\n\n{source_lang}\n{code}\n\n{target_lang}\n```\n'

    if prompt_type == 'codellama':
        prompt = f'<s>[INST] <<SYS>> You are a code translation expert. <</SYS>>\n\nTranslate the {source_lang} code below to {target_lang} and end with comment \"<END-OF-CODE>\"\n\n[/INST]\n\n{source_lang}:\n{code}\n\n{target_lang}:\n'

    if prompt_type == 'octocoder':
        prompt = f'Question: You are a code translation expert. Translate the {source_lang} code below to {target_lang} and end with comment \"<END-OF-CODE>\"\n\n{source_lang}\n{code}\n\nAnswer:\n {target_lang}\n'

    if prompt_type == 'dolphin' or prompt_type == 'mistral-hermes':
        prompt = f"""<|im_start|>system
                You are a code translation expert.<|im_end|>
                <|im_start|>user
                Can you translate the following {source_lang} code into {target_lang} and end with comment \"<END-OF-CODE>\"?
                ```{source_lang}
                {code}
                ```
                <|im_end|>
                <|im_start|>assistant
                ```{target_lan}
                """

    if prompt_type == 'starcoder':
        prompt = f"<fim_prefix>{source_lang}:\n{code}\n{target_lang}:\n<fim_suffix><fim_middle>"
                

    if prompt_type == 'solar':
        prompt = f"""<s> ### User:
        Can you translate the {source_lang} code below to {target_lang} and end with comment \"<END-OF-CODE>\"?
        ```{source_lang}
        {code}
        ```

        ### Assistant:
        Sure!
        ```{target_lang}
        """       

    if prompt_type == 'wizardcoder':
        prompt = f"""You are a code translation expert. Below is an instruction that describes a code translation task. Write a response that appropriately completes the request.

        ### Instruction:
        Write {target_lang} code that translates the following {source_lang} code and end with comment\"<END-OF-CODE>\":
        {code}

        ### Response:"""    

    if prompt_type == "deepseek":
        prompt = f'''You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
        ### Instruction:
        Translate the following {source_lang} code to {target_lang}.\n\n{source_lang}\n{code}

        ### Response:
        '''

    if prompt_type == "phi":   
         prompt = f'Instruct: You are a code translation expert. Translate the {source_lang} code below to {target_lang} and end with comment \"<END-OF-CODE>\"\n\n{source_lang}\n{code}\n\nOutput:\n {target_lang}\n'   

    if prompt_type == 'magic':
        
        prompt = f"""You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

        @@ Instruction
        You are a code translation expert. Translate the {source_lang} code below to {target_lang} and end with comment \"<END-OF-CODE>\"\n\n{source_lang}\n{code}\n'

        @@ Response
        {target_lang}
        """     

    if prompt_type == 'vicuna':
        prompt = f"""### System Prompt
                     You are a code translation expert.

                    ### User Message
                    Translate the {source_lang} code below to {target_lang} and end with comment \"<END-OF-CODE>\"\n\n{source_lang}\n{code}\n

                    ### Assistant
                   """

    return prompt


# Adopted from https://github.com/huggingface/transformers/pull/14897
class EndOfFunctionCriteria(StoppingCriteria):
    def __init__(self, start_length, eos, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_length = start_length
        self.eos = eos
        self.tokenizer = tokenizer
        self.end_length = {}

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for index, decoded_generation in enumerate(decoded_generations):
            finished = any(
                [stop_string in decoded_generation for stop_string in self.eos]
            )
            if (
                finished and index not in self.end_length
            ):  # ensures first time we see it
                for stop_string in self.eos:
                    if stop_string in decoded_generation:
                        self.end_length[index] = len(
                            input_ids[
                                index,  # get length of actual generation
                                self.start_length : -len(
                                    self.tokenizer.encode(
                                        stop_string,
                                        add_special_tokens=False,
                                        return_tensors="pt",
                                    )[0]
                                ),
                            ]
                        )
            done.append(finished)
        return all(done)


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        temperature: float = 0.8,
        conversational: bool = False,
        tensor_parallel_size: int = 1        
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature
        self.eos = EOS
        self.skip_special_tokens = False
        self.conversational = conversational
        self.tensor_parallel_size = tensor_parallel_size

    @abstractmethod
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        pass


    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

# NOTE: in order to use Gemini, the GEMINI_KEY environment variable must be set 
class GeminiDecoder(DecoderBase, ABC):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
   
        genai.configure(api_key=os.environ.get('GEMINI_KEY'))

        self.model = genai.GenerativeModel(name)

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        batch_size = min(self.batch_size, num_samples)
        if not do_sample:
            assert batch_size == 1, "Sampling only supports batch size of 1"

        outputs = []
        for _ in range(batch_size):
            try:
                response = self.model.generate_content(prompt, 
                                                    generation_config=genai.types.GenerationConfig(
                                                        # Only one candidate for now.
                                                        candidate_count=num_samples,
                                                        max_output_tokens=max_length,
                                                        temperature=self.temperature)
                                                        )
                outputs.append(response.text)                                        
            except:  
                outputs.append('GEMINI API ERROR')    
            

        return outputs    


# NOTE: in order to use Claude, the ANTHROPIC_KEY environment variable must be set 
class AnthropicDecoder(DecoderBase, ABC):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))


class AnthropicMessageDecoder(AnthropicDecoder):
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        batch_size = min(self.batch_size, num_samples)
        if not do_sample:
            assert batch_size == 1, "Sampling only supports batch size of 1"

        outputs = []
        for _ in range(batch_size):
            message = anthropic_request.make_auto_request(
                client=self.client,
                model=self.name,
                system="You are a code translation expert.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_length,
                temperature=self.temperature,
                stop_sequences=["\n```\n"],
            )
            outputs.append(message.content[0].text)

        return outputs

class VLlmDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)

        kwargs = {"tensor_parallel_size": self.tensor_parallel_size}
        if "CodeLlama" in name:
            kwargs["dtype"] = "bfloat16"
        elif "CodeBooga" in name:
            kwargs["dtype"] = "float16"
        elif "WizardCoder" in name:
            kwargs["dtype"] = "float16"
        elif "deepseek" in name:
            kwargs["dtype"] = "bfloat16"
        elif "mixtral" in name.lower():
            kwargs["dtype"] = "bfloat16"
        elif "solar" in name:
            kwargs["dtype"] = "float16"
        elif "mistral" in name.lower():
            kwargs["dtype"] = "bfloat16"
        elif "phi" in name.lower():
            kwargs["dtype"] = "float16"
            kwargs["trust_remote_code"] = True

              
        self.path = name
   
        self.llm = LLM(model=self.path, **kwargs)

        self.context_window_length = self.llm.get_tokenizer().model_max_length
        if self.context_window_length > 30000:

            # find config file 
            p = [x for x in os.listdir(cache_dir) if x.find(name.split('/')[-1])>0]
            p = [x for x in p if os.path.isdir(f'{cache_dir}/{x}')][0]
            self.path = f'{cache_dir}/{p}'

            config_path = None
            for path in glob.glob(f'{self.path}/**/config.json', recursive=True):
                config_path = path

            if config_path:
                with open(config_path) as fin:
                    config_data = json.load(fin)

                if 'n_positions' in config_data:
                    self.context_window_length = config_data['n_positions']
                elif 'max_position_embeddings' in config_data:
                    self.context_window_length = config_data['max_position_embeddings']
                else:
                    print('Model has unclear context_window_length, setting to 1024')
                    self.context_window_length = 1024
            else:
                print('Model has unclear context_window_length, setting to 1024')
                self.context_window_length = 1024   

  


    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"
        batch_size = min(self.batch_size, num_samples)

        vllm_outputs = self.llm.generate(
            [prompt] * batch_size,
            SamplingParams(
                temperature=self.temperature,
                max_tokens=min(max_length, self.context_window_length),
                top_p=0.95 if do_sample else 1.0,
                stop=self.eos,
            ),
            use_tqdm=False,
        )

        gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]
        return gen_strs


# chatml format
class ChatML(VLlmDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        kwargs["conversational"] = True
        super().__init__(name, **kwargs)
        self.eos += ["\n```"]


    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"

        return VLlmDecoder.codegen(self, prompt, do_sample, num_samples, max_length)


class HFTorchDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {
            "trust_remote_code": name
            in {
                "bigcode/santacoder",
                "Salesforce/codegen2-1B",
                "Salesforce/codegen2-3_7B",
                "Salesforce/codegen2-7B",
                "Salesforce/codegen2-16B",
                "deepseek-ai/deepseek-coder-1.3b-base",
                "deepseek-ai/deepseek-coder-6.7b-base",
                "deepseek-ai/deepseek-coder-33b-base",
                "deepseek-ai/deepseek-coder-1.3b-instruct",
                "deepseek-ai/deepseek-coder-6.7b-instruct",
                "deepseek-ai/deepseek-coder-33b-instruct"
            }
        }

        if "codegen-" in name:  # use fp16 for codegen models
            kwargs["torch_dtype"] = torch.float16
        if "codegen2-" in name:  # avoid warning of trust remote code
            kwargs["revision"] = "main"
            if "16b" in name.lower():
                kwargs["device_map"] = "auto"
        if "starcoder" in name:
            kwargs["torch_dtype"] = torch.bfloat16
        if "CodeLlama" in name:
            if "34b" in name.lower():
                kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.bfloat16
            self.skip_special_tokens = True
        if "CodeBooga" in name:
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
            self.skip_special_tokens = True
        if "Mistral-7B-codealpaca-lora" == name:
            kwargs["torch_dtype"] = torch.float16
            self.skip_special_tokens = True
        elif "Mistral" in name or "zephyr-7b-beta" in name:
            kwargs["torch_dtype"] = torch.bfloat16
        if "deepseek" in name:
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.bfloat16
            self.skip_special_tokens = True
        if "/phi" in name:
            kwargs["torch_dtype"] = torch.float16
            kwargs["trust_remote_code"] = True
            self.skip_special_tokens = True

        print(f"{kwargs} = ")

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        if name in {"StabilityAI/stablelm-base-alpha-7b"}:
            print("Switching to float16 ...")
            self.model = self.model.half()
            self.skip_special_tokens = True

        if kwargs["device_map"] != "auto":
            self.model = self.model.to(self.device)


        self.context_window_length = self.tokenizer.model_max_length
        if self.context_window_length > 1000000:
            if hasattr(self.model.config, 'n_positions'):
                self.context_window_length = self.model.config.n_positions
            elif hasattr(self.model.config, 'max_position_embeddings'):
                self.context_window_length = self.model.config.max_position_embeddings
            else:
                print('Model has unclear context_window_length, setting to 1024')
                self.context_window_length = 1024

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )
        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature

        

        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=min(max_length, self.context_window_length - len(input_tokens[0])),
            stopping_criteria=scores,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )  # remove warning
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    # could be multiple eos in outputs, better pick minimum one
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class DeepSeekInstruct(HFTorchDecoder):
    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        input_tokens = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"{prompt}",
                }
            ],
            return_tensors="pt",
        ).to(self.device)
        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature


        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=min(max_length, self.context_window_length - len(input_tokens[0])),
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
            top_k=50,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=32021,
            **kwargs,
        )  # remove warning
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        return gen_strs

# NOTE: in order to use gpt, the OPENAI_API_KEY environment variable must be set 
class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.client = openai.OpenAI()

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        batch_size = min(self.batch_size, num_samples)
        assert batch_size <= 20, "Use larger batch size could blow up the memory!"

        # construct prompt
        fmt = "text"

        ret = make_auto_request(
            self.client,
            message=prompt,
            model=self.name,
            max_tokens=self.max_length,
            temperature=self.temperature,
            n=batch_size,
            response_format={"type": fmt},
        )


        outputs = []
        for item in ret.choices:
            content = item.message.content
            # if json serializable
            if fmt == "json_object":
                try:
                    json_data = json.loads(content)
                    if json_data.get("code", None) is not None:
                        outputs.append(prompt + "\n" + json_data["code"])
                        continue

                    print(f"'code' field not found in: {json_data}")
                except Exception as e:
                    print(e)
            outputs.append(content)

        return outputs


class IncoderDecoder(HFTorchDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.infill_ph = "<|mask:0|>"
        self.extra_end = "<|mask:1|><|mask:0|>"
        self.extra_eos = [
            "<|endofmask|>",
            "<|/ file",
            "</cell>",
            "</text>",
            "</code>",
            "<|",
            "</CODE>",
        ]
        self.eos = self.eos + self.extra_eos

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        input = prompt + self.infill_ph + self.extra_end
        input_tokens = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )


        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=min(max_length, self.context_window_length - len(input_tokens[0])),
            stopping_criteria=scores,
            do_sample=do_sample,
            top_p=0.95,
            top_k=None,
            temperature=self.temperature,
            num_return_sequences=min(self.batch_size, num_samples),
            output_scores=True,
            return_dict_in_generate=True,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class Codegen2Decoder(HFTorchDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.infill_ph = "<mask_1>"
        # taken from: https://huggingface.co/Salesforce/codegen2-16B
        self.extra_end = "<|endoftext|><sep><mask_1>"
        self.extra_eos = ["<eom>"]
        self.eos = self.eos + self.extra_eos

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        input = prompt + self.infill_ph + self.extra_end
        input_tokens = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )


        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=min(max_length, self.context_window_length - len(input_tokens[0])),
            stopping_criteria=scores,
            do_sample=do_sample,
            top_p=0.95,
            top_k=None,
            temperature=self.temperature,
            output_scores=True,
            return_dict_in_generate=True,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class SantaCoder(HFTorchDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.prefix_token = "<fim-prefix>"
        self.suffix_token = "<fim-suffix>\n<fim-middle>"
        self.extra_eos = ["<|endofmask|>"]
        self.eos = self.eos + self.extra_eos

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        input = self.prefix_token + prompt + self.suffix_token
        input_tokens = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )

        if len(input_tokens[0]) >= self.context_window_length:
            outputs = []
            for _ in range(num_samples):
                outputs.append('MODEL MAX LENGTH EXCEEDED')
            return outputs

        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=min(max_length, int((self.context_window_length - len(input_tokens[0])) * 0.9)),
            stopping_criteria=scores,
            do_sample=do_sample,
            top_p=0.95,
            top_k=None,
            temperature=self.temperature,
            num_return_sequences=min(self.batch_size, num_samples),
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs,
            skip_special_tokens=self.skip_special_tokens,
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class StarCoderInfill(HFTorchDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.prefix_token = "<fim_prefix>"
        self.suffix_token = "<fim_suffix><fim_middle>"

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        input = self.prefix_token + prompt + self.suffix_token
        input_tokens = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )

        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = max(self.temperature, 1e-2)
 

        if len(input_tokens[0]) >= self.context_window_length :
            outputs = []
            for _ in range(num_samples):
                outputs.append('MODEL MAX LENGTH EXCEEDED')
            return outputs

        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=min(max_length, int((self.context_window_length  - len(input_tokens[0])) * 0.9)),
            stopping_criteria=scores,
            do_sample=do_sample,
            num_return_sequences=min(self.batch_size, num_samples),
            output_scores=True,
            return_dict_in_generate=True,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs,
            skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class CodeT5P(DecoderBase):
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert name in {
            "Salesforce/codet5p-2b",
            "Salesforce/codet5p-6b",
            "Salesforce/codet5p-16b",
            "Salesforce/instructcodet5p-16b",
        }
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            name,
            trust_remote_code=True,  # False for 220m and 770m models
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.model.to(self.device)

        self.skip_special_tokens = True

        self.context_window_length = self.tokenizer.model_max_length
        if self.context_window_length > 1000000:
            if hasattr(self.model.config, 'n_positions'):
                self.context_window_length = self.model.config.n_positions
            elif hasattr(self.model.config, 'max_position_embeddings'):
                self.context_window_length = self.model.config.max_position_embeddings
            else:
                print('Model has unclear context_window_length, setting to 1024')
                self.context_window_length = 1024 


    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        prompt = prompt.replace("    ", "\t")
        input_tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )


        max_new_tokens = min(max_length, self.context_window_length - len(input_tokens[0]))     

        while max_new_tokens > 0:
            try:
                raw_outputs = self.model.generate(
                    **input_tokens,
                    decoder_input_ids=input_tokens["input_ids"],
                    max_new_tokens=max_new_tokens,
                    stopping_criteria=scores,
                    do_sample=do_sample,
                    top_p=0.95,
                    top_k=None,
                    temperature=self.temperature,
                    output_scores=True,
                    return_dict_in_generate=True,
                    num_return_sequences=min(self.batch_size, num_samples),
                    pad_token_id=self.tokenizer.eos_token_id,
                    decoder_start_token_id=self.tokenizer.pad_token_id,
                )  # remove warning
            except RuntimeError as e:  # catch torch OOM
                if "CUDA out of memory" in str(e):
                    old_max_new_tokens = max_new_tokens
                    max_new_tokens = int(max_new_tokens * 0.8)
                    print(
                        f"OOM, reducing max_new_tokens from {old_max_new_tokens} to {max_new_tokens}"
                    )
                    continue
                else:
                    raise e

            break
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    # could be multiple eos in outputs, better pick minimum one
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index].replace("\t", "    "))
        return outputs


def make_model(name: str, batch_size: int = 1, temperature: float = 0.8, ngpus: int = 1):
    if name == "codegen-2b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="Salesforce/codegen-2B-mono",
            temperature=temperature,
        )
    elif name == "codegen-6b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="Salesforce/codegen-6B-mono",
            temperature=temperature,
        )
    elif name == "codegen-16b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="Salesforce/codegen-16B-mono",
            temperature=temperature,
        )
    elif name == "codegen2-1b":
        return Codegen2Decoder(
            batch_size=batch_size,
            name="Salesforce/codegen2-1B",
            temperature=temperature,
        )
    elif name == "codegen2-3b":
        return Codegen2Decoder(
            batch_size=batch_size,
            name="Salesforce/codegen2-3_7B",
            temperature=temperature,
        )
    elif name == "codegen2-7b":
        return Codegen2Decoder(
            batch_size=batch_size,
            name="Salesforce/codegen2-7B",
            temperature=temperature,
        )
    elif name == "codegen2-16b":
        warn(
            "codegen2-16b checkpoint is `unfinished` at this point (05/11/2023) according to their paper. "
            "So it might not make sense to use it."
        )
        return Codegen2Decoder(
            batch_size=batch_size,
            name="Salesforce/codegen2-16B",
            temperature=temperature,
        )
    elif name == "polycoder":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="NinedayWang/PolyCoder-2.7B",
            temperature=temperature,
        )
    elif name == "santacoder":
        return SantaCoder(
            batch_size=batch_size, name="bigcode/santacoder", temperature=temperature
        )
    elif name == "incoder-1b":
        return IncoderDecoder(
            batch_size=batch_size, name="facebook/incoder-1B", temperature=temperature
        )
    elif name == "incoder-6b":
        return IncoderDecoder(
            batch_size=batch_size, name="facebook/incoder-6B", temperature=temperature
        )
    elif name == "stablelm-7b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="StabilityAI/stablelm-base-alpha-7b",
            temperature=temperature,
        )
    elif name.startswith("gpt-3.5-") or name.startswith("gpt-4"):
        return OpenAIChatDecoder(
            batch_size=batch_size,
            name=name,
            temperature=temperature,
            conversational=True,
        )
    elif name.startswith("claude"):
        return AnthropicMessageDecoder(
            batch_size=batch_size,
            name=name,
            temperature=temperature,
        )
    elif name.startswith("gemini"):
        return GeminiDecoder(
            batch_size=batch_size,
            name=name,
            temperature=temperature,
        )    
    elif name == "gptneo-2b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="EleutherAI/gpt-neo-2.7B",
            temperature=temperature,
        )
    elif name == "gpt-j":
        return HFTorchDecoder(
            batch_size=batch_size, name="EleutherAI/gpt-j-6B", temperature=temperature
        )
    elif name.startswith("starcoder"):
        return VLlmDecoder(
            batch_size=batch_size,
            name=f"bigcode/{name}",
            temperature=temperature,
            tensor_parallel_size=ngpus,
        )
        # return StarCoderInfill(
        #     batch_size=batch_size, name=f"bigcode/{name}", temperature=temperature
        # )
    elif name == "codet5p-2b":
        return CodeT5P(
            batch_size=batch_size,
            name="Salesforce/codet5p-2b",
            temperature=temperature,
        )
    elif name == "codet5p-6b":
        return CodeT5P(
            batch_size=batch_size,
            name="Salesforce/codet5p-6b",
            temperature=temperature,
        )
    elif name == "codet5p-16b":
        return CodeT5P(
            batch_size=batch_size,
            name="Salesforce/codet5p-16b",
            temperature=temperature,
        )
    elif name.startswith("code-llama-"):
        assert name.endswith("b")
        nb = name.split("-")[-1]
        return VLlmDecoder(
            batch_size=batch_size,
            name=f"codellama/CodeLlama-{nb}-Instruct-hf",
            temperature=temperature,
        )
    elif name.startswith("deepseek-coder"):
        import re

        # format deepseek-coder-{nb}b*
        pattern = re.compile(r"deepseek-coder-(\d+\.?\d*)b(.*)")
        matches = pattern.findall(name)[0]
        nb = float(matches[0])
        if nb.is_integer():
            nb = int(nb)

        if "instruct" in name:
            return DeepSeekInstruct(
                batch_size=batch_size,
                name=f"deepseek-ai/deepseek-coder-{nb}b-instruct",
                temperature=temperature,
                conversational=True,
            )
        else:
            return VLlmDecoder(
                batch_size=batch_size,
                name=f"deepseek-ai/deepseek-coder-{nb}b-base",
                temperature=temperature,
                tensor_parallel_size=ngpus,
            )
    elif name == "wizardcoder-33b":
        return VLlmDecoder(
            batch_size=batch_size,
            name="WizardLM/WizardCoder-33B-V1.1",
            temperature=temperature,
            conversational=True,
            tensor_parallel_size=ngpus,
        )    
    elif name == "wizardcoder-34b":
        return VLlmDecoder(
            batch_size=batch_size,
            name="WizardLM/WizardCoder-Python-34B-V1.0",
            temperature=temperature,
            conversational=True,
            tensor_parallel_size=ngpus,
        )
    elif name == "wizardcoder-15b":
        return VLlmDecoder(
            batch_size=batch_size,
            name="WizardLM/WizardCoder-15B-V1.0",
            temperature=temperature,
            conversational=True,
            tensor_parallel_size=ngpus,
        )
    elif name == "wizardcoder-7b":
        return VLlmDecoder(
            batch_size=batch_size,
            name="WizardLM/WizardCoder-Python-7B-V1.0",
            temperature=temperature,
            conversational=True,
            tensor_parallel_size=ngpus,
        )
    elif name == "mistral-7b-codealpaca":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="Nondzu/Mistral-7B-codealpaca-lora",
            temperature=temperature,
        )
    elif name == "zephyr-7b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="HuggingFaceH4/zephyr-7b-beta",
            temperature=temperature,
        )
    elif name == "codebooga-34b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="oobabooga/CodeBooga-34B-v0.1",
            temperature=temperature,
        )
    elif name == "phind-code-llama-34b-v2":
        return VLlmDecoder(    
            batch_size=batch_size,
            name="Phind/Phind-CodeLlama-34B-v2",
            temperature=temperature,
            tensor_parallel_size=ngpus,
        )
    elif name == "mistral-7b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="mistralai/Mistral-7B-v0.1",
            temperature=temperature,
        )
    elif name == "dolphin-2.6":
        return ChatML(
            batch_size=batch_size,
            name="cognitivecomputations/dolphin-2.6-mixtral-8x7b",
            temperature=temperature,
            tensor_parallel_size=ngpus,
            # max_new_tokens=512 + 256,
        )
    elif name == "solar-10.7b-instruct":
        return ChatML(
            batch_size=batch_size,
            name="upstage/SOLAR-10.7B-Instruct-v1.0",
            temperature=temperature,
            conversational=True,
        )
    elif name == "mistral-hermes-codepro-7b":
        return ChatML(
            batch_size=batch_size,
            name="beowolx/MistralHermes-CodePro-7B-v1",
            temperature=temperature,
            # max_new_tokens=512 + 256,
        )
    elif name == "phi-2":
        return VLlmDecoder(    
            batch_size=batch_size,
            name="microsoft/phi-2",
            temperature=temperature,
            tensor_parallel_size=ngpus,
        )
    elif name == "mixtral-8x7b-instruct":
        return VLlmDecoder(
            batch_size=batch_size,
            name="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=temperature,
            tensor_parallel_size=ngpus,
        )
    elif name == "octocoder":
        return VLlmDecoder(
            batch_size=batch_size,
            name="bigcode/octocoder",
            temperature=temperature,
            tensor_parallel_size=ngpus,
        )
    elif name == "magicoder-s-ds-6.7b":
        return VLlmDecoder(
            batch_size=batch_size,
            name="ise-uiuc/Magicoder-S-DS-6.7B",
            temperature=temperature,
            tensor_parallel_size=ngpus,
        )
    elif name == "magicoder-s-cl-7b":
        return VLlmDecoder(
            batch_size=batch_size,
            name="ise-uiuc/Magicoder-S-CL-7B",
            temperature=temperature, 
            tensor_parallel_size=ngpus, 
        )

    raise ValueError(f"Invalid model name: {name}")
