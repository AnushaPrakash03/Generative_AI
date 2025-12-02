import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Controllable Text Generation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-weight: 600;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_outputs' not in st.session_state:
    st.session_state.generated_outputs = []
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = []

@st.cache_resource
def load_model():
    """Load and cache the GPT-2 model and tokenizer."""
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def generate_text(
    prompt,
    model,
    tokenizer,
    device,
    max_length=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.0,
    num_return_sequences=1
):
    """
    Generate text using the GPT-2 model with controllable parameters.
    
    Args:
        prompt: Input text prompt
        model: Pre-trained GPT-2 model
        tokenizer: GPT-2 tokenizer
        device: Torch device (CPU/GPU)
        max_length: Maximum length of generated sequence
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        repetition_penalty: Penalty for token repetition
        num_return_sequences: Number of sequences to generate
    
    Returns:
        list: Generated text sequences
    """
    # Encode input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    # Decode generated sequences
    generated_texts = []
    for sequence in output_sequences:
        text = tokenizer.decode(sequence, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts

def calculate_distinct_n(text, n=2):
    """Calculate distinct-n metric for diversity."""
    tokens = text.lower().split()
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    if len(ngrams) == 0:
        return 0.0
    
    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)
    
    return unique_ngrams / total_ngrams

def main():
    # Header
    st.title("🤖 Controllable Text Generation")
    st.markdown("""
    Explore how different parameters affect AI-generated text using GPT-2.
    Adjust temperature, top-k, and top-p to control creativity and diversity.
    """)
    
    # Load model
    with st.spinner("Loading GPT-2 model... (this may take a moment on first run)"):
        model, tokenizer, device = load_model()
    
    st.success(f"✅ Model loaded successfully! Running on: {device}")
    
    # Sidebar - Parameter Controls
    st.sidebar.header("🎛️ Control Parameters")
    
    # Quick Presets
    st.sidebar.subheader("Quick Presets")
    preset = st.sidebar.radio(
        "Choose a preset configuration:",
        ["Custom", "Conservative", "Balanced", "Creative", "Experimental"],
        help="Select a preset or choose Custom to adjust manually"
    )
    
    # Preset configurations
    presets = {
        "Conservative": {"temp": 0.3, "top_k": 20, "top_p": 0.8, "rep_penalty": 1.2},
        "Balanced": {"temp": 0.7, "top_k": 50, "top_p": 0.9, "rep_penalty": 1.1},
        "Creative": {"temp": 1.2, "top_k": 80, "top_p": 0.95, "rep_penalty": 1.0},
        "Experimental": {"temp": 1.8, "top_k": 100, "top_p": 0.98, "rep_penalty": 1.0}
    }
    
    # Set parameters based on preset or custom
    if preset != "Custom":
        config = presets[preset]
        temperature = config["temp"]
        top_k = config["top_k"]
        top_p = config["top_p"]
        repetition_penalty = config["rep_penalty"]
        st.sidebar.info(f"Using **{preset}** preset configuration")
    else:
        st.sidebar.info("Adjust parameters manually below")
        temperature = st.sidebar.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Controls randomness: Lower = focused, Higher = creative"
        )
        
        top_p = st.sidebar.slider(
            "Top-p (Nucleus Sampling)",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Samples from top tokens with cumulative probability p"
        )
        
        top_k = st.sidebar.slider(
            "Top-k Sampling",
            min_value=1,
            max_value=100,
            value=50,
            step=1,
            help="Samples from top k most likely tokens"
        )
        
        repetition_penalty = st.sidebar.slider(
            "Repetition Penalty",
            min_value=1.0,
            max_value=2.0,
            value=1.1,
            step=0.1,
            help="Penalty for repeating tokens (>1.0 discourages repetition)"
        )
    
    max_length = st.sidebar.slider(
        "Maximum Length (tokens)",
        min_value=20,
        max_value=300,
        value=100,
        step=10,
        help="Maximum length of generated text"
    )
    
    # Display current parameter values
    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Settings")
    st.sidebar.markdown(f"""
    - **Temperature:** {temperature}
    - **Top-p:** {top_p}
    - **Top-k:** {top_k}
    - **Repetition Penalty:** {repetition_penalty}
    - **Max Length:** {max_length} tokens
    """)
    
    # Main content - Two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Input Prompt")
        
        # Sample prompts
        sample_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology has advanced beyond imagination,",
            "The key to solving climate change lies in",
            "Once upon a time in a distant galaxy,",
            "The most important lesson I learned was"
        ]
        
        use_sample = st.checkbox("Use sample prompt")
        
        if use_sample:
            selected_sample = st.selectbox("Choose a sample prompt:", sample_prompts)
            prompt = st.text_area(
                "Edit or use the sample prompt:",
                value=selected_sample,
                height=100
            )
        else:
            prompt = st.text_area(
                "Enter your prompt:",
                value="The future of artificial intelligence is",
                height=100,
                help="Start typing to generate text based on your input"
            )
        
        # Generate button
        col_gen, col_clear = st.columns(2)
        
        with col_gen:
            generate_button = st.button("✨ Generate Text", use_container_width=True)
        
        with col_clear:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.generated_outputs = []
                st.session_state.comparison_results = []
                st.rerun()
        
        # Generate text
        if generate_button and prompt:
            with st.spinner("Generating text..."):
                start_time = time.time()
                
                generated_texts = generate_text(
                    prompt=prompt,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    num_return_sequences=1
                )
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                # Store in session state
                st.session_state.generated_outputs.append({
                    'text': generated_texts[0],
                    'time': generation_time,
                    'params': {
                        'temperature': temperature,
                        'top_k': top_k,
                        'top_p': top_p,
                        'repetition_penalty': repetition_penalty
                    }
                })
        
        # Display generated output
        if st.session_state.generated_outputs:
            st.header("💬 Generated Output")
            
            latest = st.session_state.generated_outputs[-1]
            
            # Output box
            st.markdown(
                f"""
                <div style='background-color: #f0f2f6; padding: 1.5rem; 
                border-radius: 10px; border-left: 5px solid #667eea;'>
                    <p style='font-size: 1.1rem; line-height: 1.6; margin: 0;'>
                        {latest['text']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Statistics
            st.subheader("📊 Generation Statistics")
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            word_count = len(latest['text'].split())
            token_count = len(tokenizer.encode(latest['text']))
            distinct_1 = calculate_distinct_n(latest['text'], n=1)
            distinct_2 = calculate_distinct_n(latest['text'], n=2)
            
            with stat_col1:
                st.markdown(
                    f"""
                    <div class='stat-card'>
                        <div class='stat-value'>{word_count}</div>
                        <div class='stat-label'>Words</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with stat_col2:
                st.markdown(
                    f"""
                    <div class='stat-card'>
                        <div class='stat-value'>{token_count}</div>
                        <div class='stat-label'>Tokens</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with stat_col3:
                st.markdown(
                    f"""
                    <div class='stat-card'>
                        <div class='stat-value'>{latest['time']:.2f}s</div>
                        <div class='stat-label'>Gen Time</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with stat_col4:
                st.markdown(
                    f"""
                    <div class='stat-card'>
                        <div class='stat-value'>{distinct_2:.2f}</div>
                        <div class='stat-label'>Diversity</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    with col2:
        st.header("ℹ️ Parameter Guide")
        
        with st.expander("🌡️ Temperature", expanded=True):
            st.markdown("""
            **Controls creativity and randomness:**
            - **Low (0.1-0.5):** Focused, deterministic
            - **Medium (0.6-1.0):** Balanced
            - **High (1.1-2.0):** Creative, diverse
            
            Lower values make the model more confident and repetitive.
            Higher values encourage exploration and creativity.
            """)
        
        with st.expander("🎯 Top-p (Nucleus Sampling)"):
            st.markdown("""
            **Samples from top tokens:**
            
            Selects from the smallest set of tokens whose cumulative 
            probability exceeds p.
            
            - **0.5:** Very focused
            - **0.9:** Recommended default
            - **0.95-1.0:** Maximum diversity
            
            More adaptive than top-k for varying confidence levels.
            """)
        
        with st.expander("🔢 Top-k Sampling"):
            st.markdown("""
            **Limits token choices:**
            
            Only considers the k most likely next tokens.
            
            - **1-20:** Very focused
            - **40-60:** Balanced
            - **80-100:** Very diverse
            
            Prevents selecting very unlikely tokens.
            """)
        
        with st.expander("🔁 Repetition Penalty"):
            st.markdown("""
            **Prevents repetition:**
            
            Penalizes tokens that have already appeared.
            
            - **1.0:** No penalty
            - **1.1-1.3:** Mild penalty (recommended)
            - **1.5+:** Strong penalty
            
            Values > 1.0 discourage repetitive text.
            """)
    
    # Comparison Feature
    st.markdown("---")
    st.header("🔄 Compare Different Settings")
    
    compare_button = st.button("Generate Comparison", use_container_width=True)
    
    if compare_button and prompt:
        with st.spinner("Generating comparisons..."):
            comparison_configs = [
                {"name": "Conservative", "temp": 0.3, "top_k": 20, "top_p": 0.8},
                {"name": "Balanced", "temp": 0.7, "top_k": 50, "top_p": 0.9},
                {"name": "Creative", "temp": 1.2, "top_k": 80, "top_p": 0.95}
            ]
            
            st.session_state.comparison_results = []
            
            for config in comparison_configs:
                texts = generate_text(
                    prompt=prompt,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=80,
                    temperature=config["temp"],
                    top_k=config["top_k"],
                    top_p=config["top_p"],
                    repetition_penalty=1.1,
                    num_return_sequences=1
                )
                
                st.session_state.comparison_results.append({
                    'config': config,
                    'text': texts[0],
                    'distinct_2': calculate_distinct_n(texts[0], n=2)
                })
    
    # Display comparison results
    if st.session_state.comparison_results:
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        
        for idx, result in enumerate(st.session_state.comparison_results):
            with [comp_col1, comp_col2, comp_col3][idx]:
                config = result['config']
                
                st.markdown(f"### {config['name']}")
                st.markdown(f"""
                **Parameters:**
                - Temp: {config['temp']}
                - Top-k: {config['top_k']}
                - Top-p: {config['top_p']}
                
                **Diversity:** {result['distinct_2']:.3f}
                """)
                
                st.markdown(
                    f"""
                    <div style='background-color: #f0f2f6; padding: 1rem; 
                    border-radius: 8px; min-height: 150px;'>
                        <p style='font-size: 0.95rem; line-height: 1.5;'>
                            {result['text']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    # Visualization: Diversity over generations
    if len(st.session_state.generated_outputs) >= 3:
        st.markdown("---")
        st.header("📈 Generation History Analysis")
        
        # Create dataframe from history
        history_data = []
        for idx, output in enumerate(st.session_state.generated_outputs[-10:], 1):
            history_data.append({
                'Generation': idx,
                'Words': len(output['text'].split()),
                'Diversity': calculate_distinct_n(output['text'], n=2),
                'Temperature': output['params']['temperature'],
                'Time (s)': output['time']
            })
        
        df_history = pd.DataFrame(history_data)
        
        # Plot diversity trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_history['Generation'],
            y=df_history['Diversity'],
            mode='lines+markers',
            name='Diversity',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Text Diversity Over Generations",
            xaxis_title="Generation Number",
            yaxis_title="Distinct-2 Score",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Controllable Text Generation Demo</strong></p>
        <p>INFO 7390 - Advances Data Science Architecture| Northeastern University</p>
        <p>Built with Streamlit and Hugging Face Transformers | Fall 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()