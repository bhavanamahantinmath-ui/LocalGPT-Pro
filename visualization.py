"""
Plotly visualization components for LocalGPT LLM X-Ray mode.
Interactive visualizers for:
- 2D PCA Token Embeddings
- Embedding Matrix Heatmap & Statistics
- Multi-Head Self-Attention Heatmap & Token Attention Flow
- Transformer Hidden State Trajectories & Depth Evolution
- Top-K Next-Token Probability Distribution
- Autoregressive Generation Step-by-Step Breakdown
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# Consistent Theme Configuration
DARK_BG = "#0b0f19"
CARD_BG = "#111827"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"
TEXT_COLOR = "#f1f5f9"
ACCENT_PRIMARY = "#6366f1"   # Indigo
ACCENT_SECONDARY = "#38bdf8" # Sky
ACCENT_SUCCESS = "#10b981"   # Emerald
ACCENT_ALERT = "#f43f5e"     # Rose

def apply_chart_theme(fig: go.Figure, title: str = "", height: int = 450) -> go.Figure:
    """Applies a consistent, polished dark theme to Plotly figures."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "font": {"size": 15, "color": TEXT_COLOR, "family": "Inter, sans-serif"},
            "x": 0.01,
            "y": 0.96
        },
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.3)",
        font={"color": "#cbd5e1", "family": "Inter, sans-serif"},
        height=height,
        margin={"l": 40, "r": 30, "t": 60 if title else 30, "b": 40},
        xaxis={
            "gridcolor": GRID_COLOR,
            "zerolinecolor": "rgba(255, 255, 255, 0.15)",
            "tickfont": {"size": 11, "color": "#94a3b8"}
        },
        yaxis={
            "gridcolor": GRID_COLOR,
            "zerolinecolor": "rgba(255, 255, 255, 0.15)",
            "tickfont": {"size": 11, "color": "#94a3b8"}
        },
        legend={"font": {"color": "#cbd5e1"}},
        hoverlabel={
            "bgcolor": "#1e293b",
            "bordercolor": "#475569",
            "font": {"color": "#ffffff", "family": "JetBrains Mono, monospace", "size": 12}
        }
    )
    return fig

# ==========================================
# 1. EMBEDDINGS VISUALIZATIONS
# ==========================================

def plot_embedding_pca(
    embeddings: np.ndarray,
    tokens_display: List[str],
    token_ids: List[int]
) -> go.Figure:
    """
    Computes 2D PCA projection of input embeddings and displays an interactive scatter plot.
    """
    n_samples = embeddings.shape[0]
    
    if n_samples < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Prompt must have at least 2 tokens for 2D PCA projection.",
            showarrow=False,
            font={"size": 14, "color": "#94a3b8"}
        )
        return apply_chart_theme(fig, "2D PCA Projection of Token Embeddings", height=380)

    n_components = min(2, n_samples, embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(embeddings)

    if n_components == 1:
        pc1 = coords[:, 0]
        pc2 = np.zeros_like(pc1)
        explained_var = [pca.explained_variance_ratio_[0] * 100, 0.0]
    else:
        pc1 = coords[:, 0]
        pc2 = coords[:, 1]
        explained_var = [pca.explained_variance_ratio_[0] * 100, pca.explained_variance_ratio_[1] * 100]

    df = pd.DataFrame({
        "Token": tokens_display,
        "Token_ID": token_ids,
        "Position": list(range(len(tokens_display))),
        "PC1": pc1,
        "PC2": pc2
    })

    fig = go.Figure()

    # Draw faint line connecting sequential tokens
    fig.add_trace(go.Scatter(
        x=df["PC1"],
        y=df["PC2"],
        mode="lines",
        line={"color": "rgba(99, 102, 241, 0.35)", "width": 1.5, "dash": "dot"},
        hoverinfo="skip",
        showlegend=False
    ))

    # Scatter points with color gradient
    fig.add_trace(go.Scatter(
        x=df["PC1"],
        y=df["PC2"],
        mode="markers+text",
        text=df["Token"],
        textposition="top center",
        textfont={"size": 12, "color": "#f8fafc", "family": "JetBrains Mono, monospace"},
        marker={
            "size": 14,
            "color": df["Position"],
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {
                "title": "Token #",
                "len": 0.8,
                "thickness": 12,
                "tickfont": {"color": "#94a3b8"}
            },
            "line": {"color": "#ffffff", "width": 1.5}
        },
        customdata=np.stack((df["Token"], df["Token_ID"], df["Position"]), axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>Position: #%{customdata[2]}<br>Token ID: %{customdata[1]}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
        showlegend=False
    ))

    title_text = f"2D PCA of Token Embeddings (PC1: {explained_var[0]:.1f}%, PC2: {explained_var[1]:.1f}% var)"
    fig = apply_chart_theme(fig, title_text, height=450)
    fig.update_xaxes(title="Principal Component 1")
    fig.update_yaxes(title="Principal Component 2")
    return fig

def plot_embedding_heatmap(
    embeddings: np.ndarray,
    tokens_display: List[str],
    max_dims: int = 64
) -> go.Figure:
    """
    Displays a heatmap showing vector values across the sequence tokens.
    """
    dims_to_show = min(max_dims, embeddings.shape[1])
    data_slice = embeddings[:, :dims_to_show]

    fig = go.Figure(data=go.Heatmap(
        z=data_slice,
        x=[f"D{i}" for i in range(dims_to_show)],
        y=[f"#{i} {t}" for i, t in enumerate(tokens_display)],
        colorscale="Blues",
        reversescale=False,
        colorbar={"title": "Value", "thickness": 12, "len": 0.8},
        hovertemplate="Token: %{y}<br>Dimension: %{x}<br>Weight: %{z:.4f}<extra></extra>"
    ))

    fig = apply_chart_theme(fig, f"Input Embedding Vectors (First {dims_to_show} of {embeddings.shape[1]} Dimensions)", height=380)
    fig.update_layout(yaxis={"autorange": "reversed"})
    fig.update_xaxes(title="Embedding Dimension Features")
    fig.update_yaxes(title="Tokens")
    return fig

def plot_embedding_stats(embeddings: np.ndarray, tokens_display: List[str]) -> go.Figure:
    """Displays L2 norm, mean, and std statistics for each token's embedding."""
    norms = np.linalg.norm(embeddings, axis=1)
    means = np.mean(embeddings, axis=1)
    stds = np.std(embeddings, axis=1)

    df = pd.DataFrame({
        "Token": [f"#{i} {t}" for i, t in enumerate(tokens_display)],
        "L2 Norm": norms,
        "Mean": means,
        "Std Dev": stds
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Token"],
        y=df["L2 Norm"],
        name="L2 Norm",
        marker_color="#6366f1",
        hovertemplate="%{x}<br>Norm: %{y:.3f}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=df["Token"],
        y=df["Std Dev"],
        name="Std Dev",
        marker_color="#38bdf8",
        hovertemplate="%{x}<br>Std: %{y:.4f}<extra></extra>"
    ))

    fig = apply_chart_theme(fig, "Token Embedding Vector Statistics (Norm & Spread)", height=350)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Tokens")
    fig.update_yaxes(title="Metric Value")
    return fig

# ==========================================
# 2. ATTENTION HEATMAP VISUALIZATION
# ==========================================

def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    tokens_display: List[str],
    layer_idx: int,
    head_idx: Optional[int] = None
) -> go.Figure:
    """
    Renders an interactive NxN attention heatmap for a specific layer and head.
    """
    labels = [f"#{i} {t}" for i, t in enumerate(tokens_display)]

    hover_matrix = []
    for r, from_tok in enumerate(labels):
        row_hovers = []
        for c, to_tok in enumerate(labels):
            val = attention_matrix[r, c]
            row_hovers.append(
                f"Query (From): <b>{from_tok}</b><br>"
                f"Key (Attends To): <b>{to_tok}</b><br>"
                f"Attention Weight: <b>{val:.4f}</b> ({val * 100:.1f}%)"
            )
        hover_matrix.append(row_hovers)

    fig = go.Figure(data=go.Heatmap(
        z=attention_matrix,
        x=labels,
        y=labels,
        text=hover_matrix,
        hoverinfo="text",
        colorscale="Magma",
        reversescale=False,
        zmin=0.0,
        zmax=max(0.1, float(np.max(attention_matrix))),
        colorbar={"title": "Attention", "thickness": 14, "len": 0.85}
    ))

    head_label = f"Head #{head_idx}" if head_idx is not None else "Mean of All Heads"
    title_text = f"Self-Attention Weights: Layer {layer_idx + 1} ({head_label})"

    fig = apply_chart_theme(fig, title_text, height=520)
    fig.update_layout(yaxis={"autorange": "reversed"})
    fig.update_xaxes(title="Key (Attended To) →", tickangle=-45)
    fig.update_yaxes(title="Query (From Token) ↓")
    return fig

def plot_last_token_attention(
    attention_matrix: np.ndarray,
    tokens_display: List[str],
    layer_idx: int,
    head_idx: Optional[int] = None
) -> go.Figure:
    """
    Shows how much attention the LAST token distributes across preceding tokens.
    """
    last_idx = len(tokens_display) - 1
    last_token_str = tokens_display[last_idx]
    weights = attention_matrix[last_idx, :]

    df = pd.DataFrame({
        "Token": [f"#{i} {t}" for i, t in enumerate(tokens_display)],
        "Weight": weights,
        "Percentage": weights * 100
    })

    fig = go.Figure(go.Bar(
        x=df["Token"],
        y=df["Percentage"],
        marker={
            "color": df["Weight"],
            "colorscale": "Plasma",
            "showscale": False
        },
        hovertemplate="Token: %{x}<br>Attention: %{y:.2f}%<extra></extra>"
    ))

    head_label = f"Head #{head_idx}" if head_idx is not None else "Mean of All Heads"
    title = f"Attention Distribution from Last Token ('{last_token_str}') in Layer {layer_idx + 1} ({head_label})"
    fig = apply_chart_theme(fig, title, height=350)
    fig.update_xaxes(title="Prompt Tokens", tickangle=-40)
    fig.update_yaxes(title="Attention (%)")
    return fig

# ==========================================
# 3. HIDDEN STATES EVOLUTION
# ==========================================

def plot_hidden_states_similarity(
    hidden_states: List[np.ndarray],
    tokens_display: List[str]
) -> go.Figure:
    """
    Visualizes cosine similarity between Layer L representations and Layer 0 across all layers.
    """
    num_layers = len(hidden_states)
    seq_len = len(tokens_display)

    layer_indices = list(range(num_layers))
    emb_layer = hidden_states[0]

    emb_norms = np.linalg.norm(emb_layer, axis=-1, keepdims=True) + 1e-8
    emb_normalized = emb_layer / emb_norms

    sim_matrix = np.zeros((num_layers, seq_len))

    for l_idx in range(num_layers):
        curr_layer = hidden_states[l_idx]
        curr_norms = np.linalg.norm(curr_layer, axis=-1, keepdims=True) + 1e-8
        curr_normalized = curr_layer / curr_norms
        sims = np.sum(emb_normalized * curr_normalized, axis=-1)
        sim_matrix[l_idx, :] = sims

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for t_idx in range(seq_len):
        tok_name = f"#{t_idx} {tokens_display[t_idx]}"
        color = colors[t_idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=layer_indices,
            y=sim_matrix[:, t_idx],
            mode="lines+markers",
            name=tok_name,
            line={"color": color, "width": 2},
            marker={"size": 5},
            hovertemplate=f"Token: <b>{tok_name}</b><br>Layer %{{x}}<br>Similarity to Input: %{{y:.3f}}<extra></extra>"
        ))

    title = "Representation Drift: Cosine Similarity with Input Embedding Across Layers"
    fig = apply_chart_theme(fig, title, height=420)
    fig.update_xaxes(title="Transformer Layer (0 = Input Embeddings, 28 = Final Layer)")
    fig.update_yaxes(title="Cosine Similarity with Layer 0", range=[-0.2, 1.05])
    return fig

def plot_token_trajectory_pca(
    hidden_states: List[np.ndarray],
    token_idx: int,
    token_label: str
) -> go.Figure:
    """
    Traces the 2D PCA trajectory of a single token vector through all layers.
    """
    token_vectors = np.stack([hs[token_idx] for hs in hidden_states], axis=0)
    
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(token_vectors)

    layers = list(range(len(hidden_states)))
    layer_names = [f"L{l}" if l > 0 else "Embed" for l in layers]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="lines",
        line={"color": "rgba(56, 189, 248, 0.5)", "width": 2, "dash": "solid"},
        hoverinfo="skip",
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers+text",
        text=layer_names,
        textposition="top right",
        textfont={"size": 10, "color": "#cbd5e1"},
        marker={
            "size": 10,
            "color": layers,
            "colorscale": "Plasma",
            "showscale": True,
            "colorbar": {"title": "Layer #", "len": 0.8, "thickness": 12},
            "line": {"color": "#ffffff", "width": 1}
        },
        hovertemplate="<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
        showlegend=False
    ))

    title = f"Hidden State Trajectory for Token '{token_label}' Through Layers"
    fig = apply_chart_theme(fig, title, height=450)
    fig.update_xaxes(title="Principal Component 1")
    fig.update_yaxes(title="Principal Component 2")
    return fig

def plot_layer_hidden_pca(
    hidden_states: List[np.ndarray],
    layer_idx: int,
    tokens_display: List[str],
    token_ids: List[int]
) -> go.Figure:
    """
    Computes 2D PCA of token representations at a specific layer.
    """
    layer_idx = min(max(0, layer_idx), len(hidden_states) - 1)
    layer_data = hidden_states[layer_idx]
    n_samples = layer_data.shape[0]

    if n_samples < 2:
        fig = go.Figure()
        fig.add_annotation(text="At least 2 tokens needed for PCA.", showarrow=False)
        return apply_chart_theme(fig, f"Layer {layer_idx} PCA", height=380)

    n_comp = min(2, n_samples, layer_data.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(layer_data)

    pc1 = coords[:, 0]
    pc2 = coords[:, 1] if n_comp > 1 else np.zeros_like(pc1)
    ev = pca.explained_variance_ratio_ * 100

    df = pd.DataFrame({
        "Token": tokens_display,
        "Token_ID": token_ids,
        "Position": list(range(len(tokens_display))),
        "PC1": pc1,
        "PC2": pc2
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["PC1"],
        y=df["PC2"],
        mode="markers+text",
        text=df["Token"],
        textposition="top center",
        marker={
            "size": 13,
            "color": df["Position"],
            "colorscale": "Turbo",
            "showscale": False,
            "line": {"color": "#ffffff", "width": 1.5}
        },
        hovertemplate="Token: <b>%{text}</b><br>ID: %{customdata}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
        customdata=df["Token_ID"]
    ))

    layer_title = f"Token Representations at Layer {layer_idx} (PC1: {ev[0]:.1f}%, PC2: {ev[1] if len(ev)>1 else 0:.1f}%)"
    fig = apply_chart_theme(fig, layer_title, height=420)
    fig.update_xaxes(title="Principal Component 1")
    fig.update_yaxes(title="Principal Component 2")
    return fig

# ==========================================
# 4. LOGITS & PROBABILITIES VISUALIZATION
# ==========================================

def plot_top_probabilities(top_candidates: List[Dict[str, Any]], top_n: int = 10) -> go.Figure:
    """
    Displays an interactive horizontal bar chart of the top next-token candidate probabilities.
    """
    slice_cands = top_candidates[:top_n]
    rev_cands = list(reversed(slice_cands))

    tokens = [c["display"] for c in rev_cands]
    probs = [c["percentage"] for c in rev_cands]
    logits = [c["logit"] for c in rev_cands]
    token_ids = [c["token_id"] for c in rev_cands]

    colors = [
        ACCENT_SUCCESS if i == len(rev_cands) - 1 else ACCENT_PRIMARY
        for i in range(len(rev_cands))
    ]

    hover_texts = [
        f"Candidate: <b>{t}</b><br>ID: {tid}<br>Probability: <b>{p:.2f}%</b><br>Logit: {l:.2f}"
        for t, tid, p, l in zip(tokens, token_ids, probs, logits)
    ]

    fig = go.Figure(go.Bar(
        x=probs,
        y=tokens,
        orientation="h",
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside",
        textfont={"color": "#f8fafc", "size": 12, "family": "JetBrains Mono, monospace"},
        marker={"color": colors, "line": {"color": "rgba(255,255,255,0.2)", "width": 1}},
        hoverinfo="text",
        hovertext=hover_texts
    ))

    title = f"Top {top_n} Next-Token Predictions (Softmax Probabilities)"
    fig = apply_chart_theme(fig, title, height=420)
    fig.update_xaxes(title="Probability (%)", range=[0, max(probs) * 1.25 if probs else 100])
    fig.update_yaxes(title="Candidate Token")
    return fig

# ==========================================
# 5. TOKEN GENERATION STEP BREAKDOWN
# ==========================================

def plot_generation_confidence_timeline(steps: List[Dict[str, Any]]) -> go.Figure:
    """
    Plots the confidence of the model for each generated token along the step timeline.
    """
    if not steps:
        fig = go.Figure()
        return apply_chart_theme(fig, "No tokens generated yet", height=320)

    step_nums = [s["step"] for s in steps]
    probs = [s["percentage"] for s in steps]
    tokens = [s["display"] for s in steps]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=step_nums,
        y=probs,
        mode="lines+markers",
        line={"color": ACCENT_SECONDARY, "width": 2.5},
        marker={"size": 8, "color": ACCENT_PRIMARY, "line": {"color": "#ffffff", "width": 1.5}},
        text=tokens,
        hovertemplate="Step %{x}: <b>%{text}</b><br>Confidence: <b>%{y:.1f}%</b><extra></extra>",
        showlegend=False
    ))

    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.25)", annotation_text="50% Confidence")

    title = "Generation Confidence per Step (Selected Token Probability)"
    fig = apply_chart_theme(fig, title, height=350)
    fig.update_xaxes(title="Generation Step #")
    fig.update_yaxes(title="Confidence (%)", range=[0, 105])
    return fig

