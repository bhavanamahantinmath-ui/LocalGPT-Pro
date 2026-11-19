"""
Model loading, multi-turn chat inference, streaming response generation,
and internal LLM X-Ray state extraction for LocalGPT.
Supports Qwen/Qwen2.5-1.5B-Instruct and Qwen/Qwen2.5-0.5B-Instruct.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Generator
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def get_optimal_device_and_dtype():
    """Detects available hardware and chooses appropriate dtype."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32

class LocalLLMEngine:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None, demo_mode: bool = False):
        self.model_name = model_name
        self.auto_device, self.auto_dtype = get_optimal_device_and_dtype()
        self.device = device or self.auto_device
        self.dtype = self.auto_dtype if self.device != "cpu" else torch.float32
        self.demo_mode = demo_mode

        self.tokenizer = None
        self.model = None
        self.config = None
        self.is_loaded = False

    def load(self):
        """Loads tokenizer and model weights with multi-tiered fallbacks."""
        if self.is_loaded and (self.model is not None or self.demo_mode):
            return self

        # 1. Load Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=True
            )
        except Exception:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            except Exception:
                try:
                    fallback_name = "Qwen/Qwen2.5-0.5B-Instruct"
                    self.tokenizer = AutoTokenizer.from_pretrained(fallback_name, local_files_only=True)
                    self.model_name = fallback_name
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        if self.demo_mode:
            self.is_loaded = True
            return self

        # 2. Load Causal LM
        loaded_successfully = False
        candidates = [self.model_name, "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
        
        for cand in candidates:
            # Try loading from local cache first
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    cand,
                    torch_dtype=self.dtype,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    attn_implementation="eager",
                    local_files_only=True
                )
                self.model_name = cand
                loaded_successfully = True
                print(f"[ModelEngine] Loaded {cand} from local cache.")
                break
            except Exception as e:
                pass

        if not loaded_successfully:
            # Try online or fallback to simulation demo mode
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    attn_implementation="eager"
                )
                loaded_successfully = True
            except Exception as e:
                print(f"[ModelEngine] Could not load real model weights ({e}). Enabling simulation mode.")
                self.demo_mode = True

        if not self.demo_mode and self.model is not None:
            if self.device != "cuda":
                self.model.to(self.device)
            self.model.eval()
            self.config = self.model.config

        self.is_loaded = True
        return self

    def get_model_metadata(self) -> Dict[str, Any]:
        """Returns architectural metadata of the model."""
        num_layers = getattr(self.config, "num_hidden_layers", 28) if self.config else 28
        num_heads = getattr(self.config, "num_attention_heads", 12) if self.config else 12
        num_kv_heads = getattr(self.config, "num_key_value_heads", num_heads) if self.config else 2
        hidden_size = getattr(self.config, "hidden_size", 1536) if self.config else 1536
        intermediate_size = getattr(self.config, "intermediate_size", 8960) if self.config else 8960
        vocab_size = getattr(self.config, "vocab_size", len(self.tokenizer) if self.tokenizer else 151936) if self.config else 151936
        
        total_params = sum(p.numel() for p in self.model.parameters()) if self.model else 1540000000

        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "total_params": total_params,
            "total_params_formatted": f"{total_params / 1e9:.2f}B" if total_params >= 1e9 else f"{total_params / 1e6:.1f}M",
            "num_layers": num_layers,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "vocab_size": vocab_size,
            "head_dim": hidden_size // num_heads if num_heads else 128,
            "is_simulation": self.demo_mode
        }

    # =========================================================================
    # Streaming Multi-Turn Text Generation
    # =========================================================================
    def format_chat_prompt(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """
        Formats conversation history into Qwen's ChatML format.
        """
        chatml_parts = []
        
        # Injected system instruction
        if system_prompt and system_prompt.strip():
            chatml_parts.append(f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>")
        elif not any(m.get("role") == "system" for m in messages):
            chatml_parts.append("<|im_start|>system\nYou are a helpful, respectful, and honest assistant.<|im_end|>")

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "").strip()
            chatml_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        chatml_parts.append("<|im_start|>assistant\n")
        return "\n".join(chatml_parts)

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> Generator[str, None, None]:
        """
        Generates response text as a real-time stream of tokens/words.
        Compatible with Streamlit's st.write_stream().
        """
        if self.demo_mode or self.model is None:
            # Fallback realistic simulated streaming
            user_last = messages[-1]["content"] if messages else "Hello"
            sim_response = (
                f"Hello! I am running in local mode. In response to '{user_last}':\n\n"
                "Artificial Intelligence and Transformer architectures operate by processing sequences "
                "of token embeddings through multi-head self-attention mechanisms and feedforward layers. "
                "Each token attends to relevant preceding tokens, building rich contextual representations.\n\n"
                "- **Multi-turn capability**: Remembers past turns.\n"
                "- **RAG**: Can integrate uploaded documents.\n"
                "- **LLM X-Ray**: Inspects inner layers, tokens, and probabilities."
            )
            words = sim_response.split(" ")
            for w in words:
                time.sleep(0.04)
                yield w + " "
            return

        # Real model inference with TextIteratorStreamer
        formatted_prompt = self.format_chat_prompt(messages, system_prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            if top_k > 0:
                gen_kwargs["top_k"] = top_k
            if top_p < 1.0:
                gen_kwargs["top_p"] = top_p

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for new_text in streamer:
            # Filter out any lingering chatml tokens if necessary
            cleaned_text = new_text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
            if cleaned_text:
                yield cleaned_text

        thread.join()

    # =========================================================================
    # LLM X-Ray Internal Forward Pass Inspection
    # =========================================================================
    @torch.no_grad()
    def inspect_inference(self, prompt: str) -> Dict[str, Any]:
        """
        Executes forward pass with output_attentions=True and output_hidden_states=True.
        Extracts embeddings, all 28 hidden state layers, attention heads, logits, and probabilities.
        """
        if self.demo_mode or self.model is None:
            # High-fidelity simulation for X-Ray
            inputs = self.tokenizer(prompt, return_tensors="pt") if self.tokenizer else {"input_ids": torch.tensor([[1, 2, 3]])}
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            hidden_size = 1536
            num_layers = 28
            num_heads = 12

            rng = np.random.RandomState(abs(hash(prompt)) % 100000)
            input_embeddings_np = rng.normal(0, 0.08, size=(seq_len, hidden_size)).astype(np.float32)

            hidden_states_list = [input_embeddings_np.copy()]
            curr_state = input_embeddings_np.copy()
            for l in range(num_layers):
                curr_state = curr_state + rng.normal(0, 0.035, size=(seq_len, hidden_size)).astype(np.float32)
                hidden_states_list.append(curr_state.copy())

            attentions_list = []
            for l in range(num_layers):
                layer_att = np.zeros((num_heads, seq_len, seq_len), dtype=np.float32)
                for h in range(num_heads):
                    raw = rng.uniform(0.1, 1.2, size=(seq_len, seq_len))
                    for r in range(seq_len):
                        for c in range(seq_len):
                            if c > r:
                                raw[r, c] = -1e9
                        raw[r, r] += 2.0
                    exp_raw = np.exp(raw - np.max(raw, axis=-1, keepdims=True))
                    layer_att[h] = exp_raw / np.sum(exp_raw, axis=-1, keepdims=True)
                attentions_list.append(layer_att)

            sample_candidates = [
                (" Artificial", 15235, 0.38, 14.8),
                (" is", 374, 0.24, 13.9),
                (" refers", 6982, 0.12, 12.8),
                (" stands", 12389, 0.08, 11.7),
                (" (", 320, 0.06, 10.9),
                (" a", 264, 0.04, 10.4),
                (" technology", 2958, 0.03, 10.1),
                (" the", 279, 0.02, 9.8),
                (" represents", 11045, 0.015, 8.5),
                (" encompasses", 45112, 0.01, 7.8),
            ]
            top_candidates = []
            for tok_str, tid, p, l in sample_candidates:
                top_candidates.append({
                    "token_id": tid,
                    "token_str": tok_str,
                    "display": tok_str if tok_str.strip() else f"[{repr(tok_str)}]",
                    "probability": p,
                    "percentage": p * 100.0,
                    "logit": l
                })

            tokens_display = []
            for tid in input_ids[0].tolist():
                d = self.tokenizer.decode([tid]) if self.tokenizer else str(tid)
                tokens_display.append(d if d.strip() else "␣")

            return {
                "prompt": prompt,
                "seq_len": seq_len,
                "input_ids": input_ids[0].tolist(),
                "tokens_display": tokens_display,
                "input_embeddings": input_embeddings_np,
                "hidden_states": hidden_states_list,
                "attentions": attentions_list,
                "top_candidates": top_candidates,
                "predicted_next_token": top_candidates[0]
            }

        # Real model inference
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        seq_len = input_ids.shape[1]

        # 1. Embeddings
        embedding_layer = self.model.get_input_embeddings()
        input_embeddings = embedding_layer(input_ids)
        input_embeddings_np = input_embeddings.squeeze(0).cpu().to(torch.float32).numpy()

        # 2. Forward pass with all hidden states and attentions
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )

        # 3. Hidden States across all layers
        hidden_states_list = []
        for hs in outputs.hidden_states:
            hidden_states_list.append(hs.squeeze(0).cpu().to(torch.float32).numpy())

        # 4. Attention Weights
        attentions_list = []
        for att in outputs.attentions:
            attentions_list.append(att.squeeze(0).cpu().to(torch.float32).numpy())

        # 5. Logits and Probabilities
        last_token_logits = outputs.logits[0, -1, :]
        probabilities = F.softmax(last_token_logits, dim=-1)

        top_k = 20
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs_np = top_probs.cpu().to(torch.float32).numpy()
        top_indices_np = top_indices.cpu().numpy()

        top_candidates = []
        for p, idx in zip(top_probs_np, top_indices_np):
            tok_str = self.tokenizer.decode([idx])
            top_candidates.append({
                "token_id": int(idx),
                "token_str": tok_str,
                "display": tok_str if tok_str.strip() else f"[{repr(tok_str)}]",
                "probability": float(p),
                "percentage": float(p * 100),
                "logit": float(last_token_logits[idx].cpu().item())
            })

        tokens_display = []
        for tid in input_ids[0].tolist():
            d = self.tokenizer.decode([tid])
            if not d:
                d = "<empty>"
            elif d == " ":
                d = "␣"
            elif d == "\n":
                d = "↵"
            tokens_display.append(d)

        return {
            "prompt": prompt,
            "seq_len": seq_len,
            "input_ids": input_ids[0].tolist(),
            "tokens_display": tokens_display,
            "input_embeddings": input_embeddings_np,
            "hidden_states": hidden_states_list,
            "attentions": attentions_list,
            "top_candidates": top_candidates,
            "predicted_next_token": top_candidates[0] if top_candidates else None
        }

    # =========================================================================
    # Step-by-Step Autoregressive Generation Trace for X-Ray
    # =========================================================================
    @torch.no_grad()
    def generate_with_trace(
        self,
        prompt: str,
        max_new_tokens: int = 25,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05
    ) -> Dict[str, Any]:
        """
        Step-by-step autoregressive generation that traces every single token step,
        logging probabilities and top alternative tokens.
        """
        if self.demo_mode or self.model is None:
            # Simulated generation trace
            demo_steps = [
                ("Artificial", 0.42, [("Artificial", 42.0), ("Machine", 25.0), ("Deep", 15.0)]),
                (" Intelligence", 0.78, [(" Intelligence", 78.0), (" learning", 12.0), (" systems", 5.0)]),
                (" (", 0.35, [(" (", 35.0), (" is", 30.0), (" refers", 18.0)]),
                ("AI", 0.92, [("AI", 92.0), ("artificial", 4.0), ("computation", 2.0)]),
                (")", 0.88, [(")", 88.0), (" is", 6.0), (",", 4.0)]),
                (" is", 0.65, [(" is", 65.0), (" enables", 15.0), (" represents", 10.0)]),
                (" the", 0.45, [(" the", 45.0), (" a", 30.0), (" an", 15.0)]),
                (" simulation", 0.38, [(" simulation", 38.0), (" field", 32.0), (" branch", 18.0)]),
                (" of", 0.85, [(" of", 85.0), (" that", 8.0), (" in", 4.0)]),
                (" human", 0.72, [(" human", 72.0), (" cognitive", 14.0), (" natural", 8.0)]),
                (" intelligence", 0.91, [(" intelligence", 91.0), (" reasoning", 5.0), (" actions", 2.0)]),
                (".", 0.82, [(".", 82.0), (" in", 10.0), (" by", 5.0)])
            ]
            steps = []
            cumulative = ""
            for i, (tok, prob, cands) in enumerate(demo_steps[:max_new_tokens]):
                cumulative += tok
                steps.append({
                    "step": i + 1,
                    "token_id": 1000 + i,
                    "token_str": tok,
                    "display": tok,
                    "probability": prob,
                    "percentage": round(prob * 100, 2),
                    "cumulative_text": cumulative,
                    "top_candidates": [{"token_str": c[0], "percentage": c[1]} for c in cands]
                })
            return {
                "prompt": prompt,
                "generated_text": cumulative,
                "total_tokens_generated": len(steps),
                "generation_time_sec": 0.8,
                "tokens_per_second": 15.0,
                "steps": steps
            }

        input_prompt = prompt
        if hasattr(self.tokenizer, "apply_chat_template") and "Instruct" in self.model_name and "<|im_start|>" not in prompt:
            try:
                input_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                input_prompt = prompt

        inputs = self.tokenizer(input_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        prompt_len = input_ids.shape[1]

        generated_steps = []
        current_input_ids = input_ids.clone()
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        start_time = time.time()

        for step in range(max_new_tokens):
            outputs = self.model(current_input_ids, return_dict=True)
            next_token_logits = outputs.logits[0, -1, :].clone()

            if repetition_penalty != 1.0 and current_input_ids.shape[1] > 0:
                for prev_id in set(current_input_ids[0].tolist()):
                    if next_token_logits[prev_id] < 0:
                        next_token_logits[prev_id] *= repetition_penalty
                    else:
                        next_token_logits[prev_id] /= repetition_penalty

            if temperature > 0:
                scaled_logits = next_token_logits / temperature
            else:
                scaled_logits = next_token_logits

            raw_probs = F.softmax(scaled_logits, dim=-1)

            if top_k > 0:
                top_k_val = min(top_k, scaled_logits.size(-1))
                indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k_val)[0][..., -1, None]
                scaled_logits[indices_to_remove] = -float("Inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                scaled_logits[indices_to_remove] = -float("Inf")

            filtered_probs = F.softmax(scaled_logits, dim=-1)

            if temperature == 0:
                next_token = torch.argmax(filtered_probs, dim=-1, keepdim=True)
            else:
                try:
                    next_token = torch.multinomial(filtered_probs, num_samples=1)
                except RuntimeError:
                    next_token = torch.argmax(filtered_probs, dim=-1, keepdim=True)

            chosen_id = int(next_token.item())
            chosen_prob = float(raw_probs[chosen_id].cpu().item())
            chosen_str = self.tokenizer.decode([chosen_id])

            step_top_probs, step_top_indices = torch.topk(raw_probs, 5)
            candidates = []
            for p, idx in zip(step_top_probs.tolist(), step_top_indices.tolist()):
                c_str = self.tokenizer.decode([idx])
                candidates.append({
                    "token_id": idx,
                    "token_str": c_str,
                    "display": c_str if c_str.strip() else f"[{repr(c_str)}]",
                    "probability": p,
                    "percentage": round(p * 100, 2)
                })

            current_input_ids = torch.cat([current_input_ids, next_token.unsqueeze(0)], dim=-1)
            cumulative_text = self.tokenizer.decode(current_input_ids[0][prompt_len:])

            generated_steps.append({
                "step": step + 1,
                "token_id": chosen_id,
                "token_str": chosen_str,
                "display": chosen_str if chosen_str.strip() else f"[{repr(chosen_str)}]",
                "probability": chosen_prob,
                "percentage": round(chosen_prob * 100, 2),
                "cumulative_text": cumulative_text,
                "top_candidates": candidates
            })

            if eos_token_id is not None and chosen_id == eos_token_id:
                break
            if chosen_str in ["<|im_end|>", "<|endoftext|>"]:
                break

        elapsed_time = time.time() - start_time
        final_response = self.tokenizer.decode(current_input_ids[0][prompt_len:], skip_special_tokens=True)
        tokens_per_second = round(len(generated_steps) / max(elapsed_time, 0.001), 2)

        return {
            "prompt": prompt,
            "generated_text": final_response,
            "total_tokens_generated": len(generated_steps),
            "generation_time_sec": round(elapsed_time, 2),
            "tokens_per_second": tokens_per_second,
            "steps": generated_steps
        }

