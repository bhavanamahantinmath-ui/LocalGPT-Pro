"""
LLM X-Ray inspection panel for LocalGPT.
Provides detailed interactive inspection into:
1. Tokens & Tokenization
2. Input Embeddings & 2D PCA
3. Transformer Layers & Depth Dynamics
4. Multi-Head Self-Attention Matrices
5. Hidden State Trajectories
6. Next-Token Logits & Softmax Probabilities
7. Autoregressive Generation Step-by-Step Breakdown
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from tokenizer import TokenizerInspector
import visualization as viz

class LLMXRayInspector:
    def __init__(self, engine):
        self.engine = engine
        self.tokenizer_inspector = TokenizerInspector(engine.tokenizer) if engine.tokenizer else None

    def render(self, text_to_inspect: str, trace_data: Optional[Dict[str, Any]] = None):
        """
        Renders the full interactive 6-tab X-Ray inspection panel in Streamlit.
        """
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(56,189,248,0.08)); 
                    border: 1px solid rgba(99,102,241,0.3); border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h3 style="margin: 0; color: #f8fafc; font-size: 1.25rem;">🔬 LLM X-Ray Inspection Suite</h3>
                    <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.85rem;">
                        Hook-free neural inspection into internal representations, attention weights, and logits.
                    </p>
                </div>
                <div style="text-align: right;">
                    <span style="background: #1e293b; color: #38bdf8; font-size: 0.75rem; padding: 4px 10px; border-radius: 20px; font-weight: 600;">
                        Model: Qwen2.5-Instruct
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not text_to_inspect or not text_to_inspect.strip():
            st.info("No prompt or response selected for X-Ray inspection. Enter a message or select a turn to inspect.")
            return

        with st.spinner("Executing neural forward pass for X-Ray state extraction..."):
            xray_data = self.engine.inspect_inference(text_to_inspect)

        # Tab navigation
        tab_tokens, tab_embed, tab_layers, tab_attn, tab_hidden, tab_logits = st.tabs([
            "🔤 Tokens",
            "🌐 Embeddings",
            "🥞 Layers",
            "🎯 Attention",
            "🧬 Hidden States",
            "📊 Logits & Probabilities"
        ])

        tokens_display = xray_data["tokens_display"]
        input_ids = xray_data["input_ids"]
        seq_len = xray_data["seq_len"]

        # =====================================================================
        # 1. TOKENS TAB
        # =====================================================================
        with tab_tokens:
            st.markdown("#### 🔤 Tokenization Breakdown")
            
            # Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            char_count = len(text_to_inspect)
            ratio = round(char_count / max(seq_len, 1), 2)
            vocab_size = getattr(self.engine.tokenizer, "vocab_size", 151936) if self.engine.tokenizer else 151936

            with col1:
                st.metric("Total Tokens", seq_len)
            with col2:
                st.metric("Character Count", char_count)
            with col3:
                st.metric("Compression Ratio", f"{ratio} ch/tok")
            with col4:
                st.metric("Vocab Size", f"{vocab_size:,}")

            st.markdown("##### Visual Token Stream (Hover for details)")
            if self.tokenizer_inspector:
                inspection = self.tokenizer_inspector.inspect(text_to_inspect)
                st.markdown(
                    TokenizerInspector.render_tokens_html(inspection["tokens"]),
                    unsafe_allow_html=True
                )

            st.markdown("##### Detailed Token Mapping Table")
            token_rows = []
            for i, (tid, dtok) in enumerate(zip(input_ids, tokens_display)):
                raw_tok = self.engine.tokenizer.convert_ids_to_tokens(tid) if self.engine.tokenizer else str(dtok)
                token_rows.append({
                    "Index": i,
                    "Token ID": tid,
                    "Display String": dtok,
                    "Raw Token": raw_tok,
                    "Byte Length": len(str(dtok).encode("utf-8"))
                })
            st.dataframe(pd.DataFrame(token_rows), use_container_width=True, height=220)

        # =====================================================================
        # 2. EMBEDDINGS TAB
        # =====================================================================
        with tab_embed:
            st.markdown("#### 🌐 Input Embeddings Space")
            embeddings = xray_data["input_embeddings"]
            
            st.caption(f"Embedding Matrix Dimensions: **{embeddings.shape[0]} tokens × {embeddings.shape[1]} features**")

            # 2D PCA Projection
            fig_pca = viz.plot_embedding_pca(embeddings, tokens_display, input_ids)
            st.plotly_chart(fig_pca, use_container_width=True)

            col_heat, col_stats = st.columns([1, 1])
            with col_heat:
                fig_heat = viz.plot_embedding_heatmap(embeddings, tokens_display, max_dims=48)
                st.plotly_chart(fig_heat, use_container_width=True)
            with col_stats:
                fig_stats = viz.plot_embedding_stats(embeddings, tokens_display)
                st.plotly_chart(fig_stats, use_container_width=True)

        # =====================================================================
        # 3. LAYERS TAB
        # =====================================================================
        with tab_layers:
            st.markdown("#### 🥞 Transformer Depth Dynamics (All Layers)")
            hidden_states = xray_data["hidden_states"]
            num_layers = len(hidden_states) - 1

            st.caption(
                f"Tracking contextual evolution across **{num_layers} Transformer layers** + initial embedding."
            )

            # Representation Drift
            fig_drift = viz.plot_hidden_states_similarity(hidden_states, tokens_display)
            st.plotly_chart(fig_drift, use_container_width=True)

            # Layer-Specific PCA
            st.markdown("##### Layer-by-Layer Token Geometry")
            selected_layer = st.slider(
                "Select Transformer Layer for 2D PCA Inspection",
                min_value=0,
                max_value=num_layers,
                value=num_layers // 2,
                key="xray_layer_pca_slider"
            )
            fig_layer_pca = viz.plot_layer_hidden_pca(
                hidden_states=hidden_states,
                layer_idx=selected_layer,
                tokens_display=tokens_display,
                token_ids=input_ids
            )
            st.plotly_chart(fig_layer_pca, use_container_width=True)

        # =====================================================================
        # 4. ATTENTION TAB
        # =====================================================================
        with tab_attn:
            st.markdown("#### 🎯 Multi-Head Self-Attention")
            attentions = xray_data["attentions"]
            num_attn_layers = len(attentions)

            if num_attn_layers > 0:
                num_heads = attentions[0].shape[0]

                col_l, col_h = st.columns([1, 1])
                with col_l:
                    chosen_layer = st.slider(
                        "Attention Layer",
                        min_value=1,
                        max_value=num_attn_layers,
                        value=min(12, num_attn_layers),
                        key="xray_attn_layer_slider"
                    ) - 1
                with col_h:
                    head_options = ["Average Across All Heads"] + [f"Head {i}" for i in range(num_heads)]
                    chosen_head_str = st.selectbox(
                        "Attention Head",
                        head_options,
                        index=0,
                        key="xray_attn_head_select"
                    )

                if chosen_head_str == "Average Across All Heads":
                    head_idx = None
                    matrix = np.mean(attentions[chosen_layer], axis=0)
                else:
                    head_idx = int(chosen_head_str.split(" ")[1])
                    matrix = attentions[chosen_layer][head_idx]

                fig_attn = viz.plot_attention_heatmap(matrix, tokens_display, chosen_layer, head_idx)
                st.plotly_chart(fig_attn, use_container_width=True)

                fig_last_attn = viz.plot_last_token_attention(matrix, tokens_display, chosen_layer, head_idx)
                st.plotly_chart(fig_last_attn, use_container_width=True)
            else:
                st.warning("Attention matrices were not captured for this forward pass.")

        # =====================================================================
        # 5. HIDDEN STATES TAB
        # =====================================================================
        with tab_hidden:
            st.markdown("#### 🧬 Single Token Trajectory Through Depth")
            hidden_states = xray_data["hidden_states"]

            tok_choices = [f"#{i}: {t}" for i, t in enumerate(tokens_display)]
            selected_tok_str = st.selectbox(
                "Select a token to track its path through all 28 layers:",
                tok_choices,
                index=len(tok_choices) - 1,
                key="xray_tok_traj_select"
            )
            selected_tok_idx = int(selected_tok_str.split(":")[0].replace("#", ""))

            fig_traj = viz.plot_token_trajectory_pca(
                hidden_states=hidden_states,
                token_idx=selected_tok_idx,
                token_label=tokens_display[selected_tok_idx]
            )
            st.plotly_chart(fig_traj, use_container_width=True)

        # =====================================================================
        # 6. LOGITS & PROBABILITIES TAB
        # =====================================================================
        with tab_logits:
            st.markdown("#### 📊 Next Token Prediction Distribution")
            top_cands = xray_data["top_candidates"]

            if top_cands:
                top_1 = top_cands[0]
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    st.metric("Most Likely Next Token", f"'{top_1['display']}'", f"{top_1['percentage']:.2f}%")
                with col_p2:
                    st.metric("Raw Logit Score", f"{top_1['logit']:.2f}")
                with col_p3:
                    entropy = -sum((c["probability"] * np.log(max(c["probability"], 1e-9))) for c in top_cands)
                    st.metric("Prediction Entropy", f"{entropy:.2f} nats")

                fig_probs = viz.plot_top_probabilities(top_cands, top_n=12)
                st.plotly_chart(fig_probs, use_container_width=True)

            # Autoregressive Generation Step Trace if available
            if trace_data and trace_data.get("steps"):
                st.markdown("##### ⏱️ Autoregressive Generation Timeline")
                fig_timeline = viz.plot_generation_confidence_timeline(trace_data["steps"])
                st.plotly_chart(fig_timeline, use_container_width=True)

                st.markdown("##### Step-by-Step Candidate Breakdown")
                trace_rows = []
                for s in trace_data["steps"]:
                    runner_ups = ", ".join([f"{c['token_str']} ({c.get('percentage', 0)}%)" for c in s.get("top_candidates", [])[1:4]])
                    trace_rows.append({
                        "Step": s["step"],
                        "Chosen Token": s["display"],
                        "Confidence": f"{s['percentage']}%",
                        "Runner-Up Candidates": runner_ups,
                        "Cumulative Text": s["cumulative_text"]
                    })
                st.dataframe(pd.DataFrame(trace_rows), use_container_width=True, height=250)

