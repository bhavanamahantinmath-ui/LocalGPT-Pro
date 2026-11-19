"""
Tokenizer Inspector and representation utilities for LocalGPT.
Extracts token IDs, strings, display forms, byte representations,
and generates beautiful pastel HTML token badges for the UI.
"""

from typing import Dict, Any, List
import html

class TokenizerInspector:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def inspect(self, text: str) -> Dict[str, Any]:
        """
        Tokenizes the input text and returns rich token-level details.
        """
        encoding = self.tokenizer(text, return_tensors="pt")
        input_ids = encoding["input_ids"][0].tolist()
        
        try:
            raw_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        except Exception:
            raw_tokens = [self.tokenizer.decode([tid]) for tid in input_ids]
        
        token_items = []
        for idx, (token_id, raw_tok) in enumerate(zip(input_ids, raw_tokens)):
            try:
                decoded = self.tokenizer.decode([token_id])
            except Exception:
                decoded = str(raw_tok)

            display_tok = str(raw_tok)
            is_space_prefix = False
            if display_tok.startswith("Ġ"):
                display_tok = " " + display_tok[1:]
                is_space_prefix = True
            elif display_tok.startswith(" "):
                display_tok = " " + display_tok[1:]
                is_space_prefix = True
            elif display_tok.startswith("<0x"):
                display_tok = decoded

            token_items.append({
                "index": idx,
                "token_id": int(token_id),
                "raw_token": str(raw_tok),
                "decoded": decoded,
                "display": display_tok,
                "is_space_prefix": is_space_prefix,
                "length": len(decoded)
            })

        char_count = len(text)
        token_count = len(input_ids)
        ratio = round(char_count / max(token_count, 1), 2)

        return {
            "text": text,
            "input_ids": input_ids,
            "tensor_input_ids": encoding["input_ids"],
            "attention_mask": encoding.get("attention_mask", None),
            "tokens": token_items,
            "num_tokens": token_count,
            "num_characters": char_count,
            "compression_ratio": ratio,
            "vocab_size": getattr(self.tokenizer, "vocab_size", len(self.tokenizer))
        }

    @staticmethod
    def render_tokens_html(token_items: List[Dict[str, Any]]) -> str:
        """
        Generates visually styled HTML badges with alternating pastel colors
        and rich tooltips for Streamlit display.
        """
        badge_palette = [
            ("rgba(99, 102, 241, 0.15)", "#818cf8", "rgba(99, 102, 241, 0.4)"),  # Indigo
            ("rgba(236, 72, 153, 0.15)", "#f472b6", "rgba(236, 72, 153, 0.4)"),  # Pink
            ("rgba(16, 185, 129, 0.15)", "#34d399", "rgba(16, 185, 129, 0.4)"),  # Emerald
            ("rgba(245, 158, 11, 0.15)", "#fbbf24", "rgba(245, 158, 11, 0.4)"),  # Amber
            ("rgba(14, 165, 233, 0.15)", "#38bdf8", "rgba(14, 165, 233, 0.4)"),  # Sky
            ("rgba(168, 85, 247, 0.15)", "#c084fc", "rgba(168, 85, 247, 0.4)"),  # Purple
        ]

        html_badges = []
        for item in token_items:
            idx = item["index"]
            bg, text_color, border = badge_palette[idx % len(badge_palette)]
            raw_display = item["display"]
            
            prefix_indicator = ""
            if item["is_space_prefix"]:
                prefix_indicator = "<span style='opacity:0.4; font-size:10px; margin-right:1px;'>␣</span>"

            safe_text = html.escape(raw_display.strip() if item["is_space_prefix"] else raw_display)
            if not safe_text and item["is_space_prefix"]:
                safe_text = "space"

            tooltip_content = (
                f"Index: {idx}&#10;"
                f"Token ID: {item['token_id']}&#10;"
                f"Raw: {repr(item['raw_token'])}&#10;"
                f"Decoded: {repr(item['decoded'])}"
            )

            badge_html = (
                f'<span title="{tooltip_content}" style="'
                f'display: inline-flex; align-items: center; margin: 3px 2px; padding: 3px 8px; '
                f'background: {bg}; border: 1px solid {border}; border-radius: 6px; '
                f'font-family: \'JetBrains Mono\', Consolas, monospace; font-size: 13px; color: {text_color}; '
                f'cursor: pointer; transition: all 0.2s ease;'
                f'">'
                f'{prefix_indicator}'
                f'<span style="font-weight: 500;">{safe_text}</span>'
                f'<span style="font-size: 9px; opacity: 0.55; margin-left: 5px;">#{item["token_id"]}</span>'
                f'</span>'
            )
            html_badges.append(badge_html)

        container_html = (
            f'<div style="'
            f'background: #111827; border: 1px solid #1f2937; border-radius: 8px; '
            f'padding: 14px; line-height: 1.8; max-height: 280px; overflow-y: auto;'
            f'">'
            f'{"".join(html_badges)}'
            f'</div>'
        )
        return container_html

