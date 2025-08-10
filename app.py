import streamlit as st
import pandas as pd
import google.generativeai as genai
from google.generativeai import caching
import time
import datetime
import hashlib
import json
import io
import PyPDF2
import docx
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
import re
import numpy as np

# Configure page
st.set_page_config(
    page_title="Hedge Fund Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better readability
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .analysis-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .performance-table {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .highlight-text {
        background-color: #fff3cd;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .stExpander > div:first-child {
        background-color: #f0f2f6;
    }
    .recommendation-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class FundAnalysis:
    """Data class for comprehensive fund analysis results"""
    executive_summary: str
    key_themes: List[str]
    competitive_edge: str
    key_people: List[Dict[str, str]]
    pros: List[str]
    cons: List[str]
    risk_assessment: str
    return_analysis: str
    recommendation: str
    performance_insights: Optional[str] = None
    benchmark_comparison: Optional[str] = None

class HedgeFundAnalyzer:
    def __init__(self, api_key: str, model_name: str = 'gemini-2.0-flash-exp'):
        """Initialize the analyzer with advanced Gemini models"""
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.cached_contexts = {}
        
        # Model configurations for different analysis types
        self.model_configs = {
            'gemini-2.0-flash-exp': {
                'name': 'Gemini 2.0 Flash (Experimental)',
                'description': 'Latest experimental model with enhanced reasoning',
                'best_for': 'Complex financial analysis, latest capabilities'
            },
            'gemini-1.5-pro-002': {
                'name': 'Gemini 1.5 Pro (Latest)',
                'description': 'Most stable advanced model',
                'best_for': 'Production analysis, consistent results'
            },
            'gemini-1.5-flash-002': {
                'name': 'Gemini 1.5 Flash (Latest)',
                'description': 'Faster analysis with good quality',
                'best_for': 'Quick analysis, cost efficiency'
            },
            'gemini-exp-1121': {
                'name': 'Gemini Experimental 1121',
                'description': 'Advanced experimental model',
                'best_for': 'Cutting-edge analysis capabilities'
            }
        }
    
    def create_context_cache(self, content: str, cache_id: str) -> str:
        """Create or retrieve cached context for advanced Gemini models"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        cache_key = f"{cache_id}_{content_hash}_{self.model_name}"
        
        if cache_key in self.cached_contexts:
            return self.cached_contexts[cache_key]
            
        try:
            # Use model-specific caching with longer TTL for advanced models
            cache_model = f'models/{self.model_name}'
            if 'exp' in self.model_name or '2.0' in self.model_name:
                # Advanced models get longer cache time
                ttl_hours = 2
            else:
                ttl_hours = 1
                
            cache = caching.CachedContent.create(
                model=cache_model,
                display_name=f'advanced_fund_analysis_{cache_id}_{self.model_name}',
                contents=[content],
                ttl=datetime.timedelta(hours=ttl_hours)
            )
            self.cached_contexts[cache_key] = cache.name
            return cache.name
        except Exception as e:
            st.warning(f"Context caching not available for {self.model_name}: {e}")
            return None
    
    def extract_text_from_pdf(self, file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"PDF extraction failed: {e}")
            return ""
    
    def extract_text_from_docx(self, file) -> str:
        """Extract text from Word document"""
        try:
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Word document extraction failed: {e}")
            return ""

    def process_csv_excel(self, file) -> tuple[str, pd.DataFrame]:
        """Process CSV/Excel files for performance data"""
        try:
            file_extension = file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file)
            else:
                return "", pd.DataFrame()
            
            # Basic data summary
            summary = f"""
Performance Data Summary for {file.name}:
- Rows: {len(df)}
- Columns: {len(df.columns)}
- Column Names: {', '.join(df.columns.tolist())}

Sample Data (first 3 rows):
{df.head(3).to_string()}

Data Types:
{df.dtypes.to_string()}
            """
            
            # Try to identify performance metrics
            performance_text = self.analyze_performance_data(df)
            
            return summary + "\n\n" + performance_text, df
            
        except Exception as e:
            st.error(f"CSV/Excel processing failed: {e}")
            return "", pd.DataFrame()
    
    def analyze_performance_data(self, df: pd.DataFrame) -> str:
        """Analyze performance data and generate insights"""
        insights = []
        
        # Look for common performance columns
        perf_columns = []
        date_columns = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['return', 'performance', 'nav', 'price', 'yield', 'gain', 'loss']):
                perf_columns.append(col)
            if any(keyword in col_lower for keyword in ['date', 'month', 'year', 'period', 'time']):
                date_columns.append(col)
        
        if perf_columns:
            insights.append(f"Performance Metrics Identified: {', '.join(perf_columns)}")
            
            # Calculate basic statistics for performance columns
            for col in perf_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_stats = df[col].describe()
                    insights.append(f"""
{col} Statistics:
- Mean: {col_stats['mean']:.2f}%
- Std Dev: {col_stats['std']:.2f}%
- Min: {col_stats['min']:.2f}%
- Max: {col_stats['max']:.2f}%
                    """)
        
        if date_columns:
            insights.append(f"Date Columns Found: {', '.join(date_columns)}")
            
            # Try to determine time range
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_range = f"From {df[col].min()} to {df[col].max()}"
                    insights.append(f"Time Period ({col}): {date_range}")
                except:
                    pass
        
        return "\n".join(insights) if insights else "No specific performance metrics identified in the data."
    
    def process_uploaded_files(self, uploaded_files) -> tuple[str, List[pd.DataFrame]]:
        """Process and combine text from all uploaded files, including performance data"""
        combined_text = ""
        performance_dataframes = []
        
        for file in uploaded_files:
            file_extension = file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                text = self.extract_text_from_pdf(file)
                combined_text += f"\n\n--- Content from {file.name} ---\n\n{text}"
                
            elif file_extension == 'docx':
                text = self.extract_text_from_docx(file)
                combined_text += f"\n\n--- Content from {file.name} ---\n\n{text}"
                
            elif file_extension == 'txt':
                text = str(file.read(), "utf-8")
                combined_text += f"\n\n--- Content from {file.name} ---\n\n{text}"
                
            elif file_extension in ['csv', 'xlsx', 'xls']:
                # Process performance data files
                summary_text, df = self.process_csv_excel(file)
                if not df.empty:
                    performance_dataframes.append(df)
                    combined_text += f"\n\n--- Performance Data from {file.name} ---\n\n{summary_text}"
                
            else:
                st.warning(f"Unsupported file type: {file.name}")
                continue
        
        return combined_text, performance_dataframes
    
    def analyze_fund(self, content: str, performance_data: List[pd.DataFrame] = None) -> FundAnalysis:
        """Analyze fund using advanced Gemini AI models with performance data integration"""
        
        # Create context cache
        cache_name = self.create_context_cache(content, "advanced_fund_analysis")
        
        # Enhanced analysis prompt for advanced models with performance data
        analysis_prompt = f"""
        As a senior investment partner with 20+ years of hedge fund experience, perform a comprehensive 
        institutional-grade analysis of the provided fund materials. 
        
        {'PERFORMANCE DATA AVAILABLE: Incorporate the performance data analysis in your evaluation.' if performance_data else ''}
        
        Use advanced reasoning to identify subtle risks and opportunities. Consider market cycles, 
        regime changes, and systematic risks.
        
        Provide analysis in this JSON format:
        
        {{
            "executive_summary": "Professional 4-5 paragraph summary covering strategy, performance analysis, market positioning, team assessment, and investment thesis. Include quantitative insights from performance data if available.",
            
            "key_themes": [
                "Strategic theme with market context and competitive positioning",
                "Performance theme incorporating historical track record analysis", 
                "Risk management theme with portfolio construction insights",
                "Market opportunity theme with timing and cycle considerations",
                "Operational excellence theme with scalability assessment"
            ],
            
            "competitive_edge": "Detailed analysis of sustainable competitive advantages, barriers to entry, and differentiation. Assess whether advantages are structural, informational, or execution-based with supporting evidence.",
            
            "key_people": [
                {{
                    "name": "Full Name",
                    "role": "Specific Title", 
                    "background": "Comprehensive background including track record, previous funds, education, and key achievements",
                    "risk_assessment": "Key-man risk evaluation and succession planning assessment"
                }}
            ],
            
            "pros": [
                "Strategic Advantage: Detailed competitive positioning with sustainability analysis",
                "Performance Excellence: Track record strength with risk-adjusted metrics", 
                "Team Quality: Human capital advantages with retention strategies",
                "Process Sophistication: Investment process and risk management excellence",
                "Market Positioning: Timing and opportunity assessment with catalyst analysis",
                "Operational Infrastructure: Technology, systems, and scalability advantages"
            ],
            
            "cons": [
                "Performance Risk: Historical volatility or drawdown concerns with impact analysis",
                "Competitive Pressure: Market saturation or strategy commoditization risks",
                "Operational Constraints: Capacity limitations or infrastructure gaps", 
                "Market Sensitivity: Cycle dependency or regime change vulnerabilities",
                "Liquidity Concerns: Redemption restrictions or portfolio liquidity mismatches"
            ],
            
            "risk_assessment": "Multi-dimensional risk analysis covering Market Risk (beta exposures, factor sensitivities), Credit Risk (counterparty exposure), Operational Risk (key-man, systems), Structural Risk (leverage, liquidity terms), and Performance Risk (volatility, drawdowns, correlation). Include stress testing insights.",
            
            "return_analysis": "Sophisticated return evaluation including risk-adjusted metrics (Sharpe, Sortino, Calmar ratios), drawdown analysis, return consistency, factor attribution, benchmark comparisons, and forward-looking return expectations with confidence intervals.",
            
            "recommendation": "Investment committee recommendation with allocation sizing, timing considerations, portfolio fit analysis, risk budget impact, conviction level (1-10), and key monitoring metrics. Include implementation pathway and due diligence priorities.",
            
            "performance_insights": "Detailed analysis of historical performance data including return patterns, volatility analysis, correlation studies, and benchmark comparisons. Identify performance drivers and risk factors.",
            
            "benchmark_comparison": "Comparative analysis against relevant benchmarks and peer funds. Assess relative performance, risk metrics, and competitive positioning in the strategy space."
        }}
        
        Analysis Framework:
        - Alpha Generation: Sustainability and decay analysis
        - Risk Management: Portfolio construction and downside protection
        - Performance Attribution: Factor exposures and return drivers  
        - Market Positioning: Competitive landscape and differentiation
        - Operational Excellence: Infrastructure and process sophistication
        - Team Assessment: Experience, stability, and succession planning
        
        Provide actionable insights for institutional investment decisions.
        """
        
        try:
            # Enhanced generation config for advanced models
            generation_config = {
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json"
            }
            
            if cache_name:
                model_with_cache = genai.GenerativeModel(
                    self.model_name, 
                    cached_content=cache_name,
                    generation_config=generation_config
                )
                response = model_with_cache.generate_content(analysis_prompt)
            else:
                model_with_config = genai.GenerativeModel(
                    self.model_name,
                    generation_config=generation_config
                )
                full_prompt = f"{analysis_prompt}\n\nFund Materials:\n{content}"
                response = model_with_config.generate_content(full_prompt)
            
            # Enhanced JSON parsing
            response_text = response.text.strip()
            
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            try:
                analysis_data = json.loads(response_text)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")
            
            return FundAnalysis(**analysis_data)
            
        except json.JSONDecodeError as e:
            st.error(f"AI response parsing failed: {e}")
            st.error(f"Response preview: {response_text[:500]}...")
            return None
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return None
def create_performance_visualization(performance_data: List[pd.DataFrame]) -> go.Figure:
    """Create comprehensive performance visualization from uploaded data"""
    fig = go.Figure()
    
    if not performance_data:
        # Create sample visualization when no data is available
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='M')
        sample_returns = np.random.normal(0.008, 0.04, len(dates))
        cumulative_returns = (1 + pd.Series(sample_returns)).cumprod() - 1
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_returns * 100,
            mode='lines',
            name='Sample Fund Performance',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add benchmark
        benchmark_returns = np.random.normal(0.006, 0.03, len(dates))
        benchmark_cumulative = (1 + pd.Series(benchmark_returns)).cumprod() - 1
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark_cumulative * 100,
            mode='lines',
            name='Benchmark',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
    else:
        # Process actual uploaded performance data
        for i, df in enumerate(performance_data):
            # Try to identify date and performance columns
            date_col = None
            perf_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'date' in col_lower or 'month' in col_lower or 'period' in col_lower:
                    date_col = col
                if 'return' in col_lower or 'performance' in col_lower or 'nav' in col_lower:
                    perf_col = col
                    
            if date_col and perf_col:
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    fig.add_trace(go.Scatter(
                        x=df[date_col],
                        y=df[perf_col],
                        mode='lines+markers',
                        name=f'Fund Performance {i+1}',
                        line=dict(width=2)
                    ))
                except Exception as e:
                    st.warning(f"Could not plot data from file {i+1}: {e}")
    
    fig.update_layout(
        title='Fund Performance Analysis',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_risk_metrics_chart(performance_data: List[pd.DataFrame]) -> go.Figure:
    """Create risk metrics visualization"""
    fig = go.Figure()
    
    if performance_data:
        # Calculate risk metrics from actual data
        metrics_data = []
        
        for i, df in enumerate(performance_data):
            # Find performance column
            perf_col = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['return', 'performance', 'nav']):
                    perf_col = col
                    break
            
            if perf_col and pd.api.types.is_numeric_dtype(df[perf_col]):
                returns = df[perf_col].dropna()
                
                # Calculate metrics
                volatility = returns.std() * np.sqrt(12) * 100  # Annualized
                sharpe = (returns.mean() * 12) / (returns.std() * np.sqrt(12)) if returns.std() > 0 else 0
                max_dd = (returns.cumsum() - returns.cumsum().expanding().max()).min()
                
                metrics_data.append({
                    'Fund': f'Fund {i+1}',
                    'Volatility (%)': volatility,
                    'Sharpe Ratio': sharpe,
                    'Max Drawdown (%)': abs(max_dd) * 100
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            
            # Create grouped bar chart
            fig.add_trace(go.Bar(
                x=metrics_df['Fund'],
                y=metrics_df['Volatility (%)'],
                name='Volatility (%)',
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                x=metrics_df['Fund'],
                y=metrics_df['Sharpe Ratio'],
                name='Sharpe Ratio',
                marker_color='lightgreen',
                yaxis='y2'
            ))
    else:
        # Sample risk metrics
        sample_funds = ['Fund A', 'Fund B', 'Benchmark']
        volatilities = [12.5, 15.2, 16.8]
        sharpe_ratios = [1.2, 0.9, 0.7]
        
        fig.add_trace(go.Bar(
            x=sample_funds,
            y=volatilities,
            name='Volatility (%)',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=sample_funds,
            y=sharpe_ratios,
            name='Sharpe Ratio',
            marker_color='lightgreen',
            yaxis='y2'
        ))
    
    fig.update_layout(
        title='Risk Metrics Comparison',
        xaxis_title='Funds',
        yaxis=dict(title='Volatility (%)', side='left'),
        yaxis2=dict(title='Sharpe Ratio', side='right', overlaying='y'),
        height=400,
        barmode='group'
    )
    
    return fig
def main():
    # Enhanced header with better styling
    st.markdown('<h1 class="main-header">🏦 Professional Fund Analysis Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Institutional-Grade Investment Analysis • Powered by Advanced AI</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### 🚀 Configuration Panel")
        
        # API Key input
        api_key = st.text_input(
            "🔑 Gemini API Key", 
            type="password",
            help="Enter your Google Gemini API key for advanced analysis",
            placeholder="Enter API key here..."
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your Gemini API key to begin analysis")
            st.info("💡 Get your API key from Google AI Studio")
            return
        
        # Model selection with clearer descriptions
        st.markdown("### 🧠 AI Model Selection")
        
        model_options = {
            'gemini-2.0-flash-exp': '🌟 Gemini 2.0 Flash (Latest) - Most Advanced',
            'gemini-exp-1121': '🔬 Gemini Experimental - Enhanced Reasoning',
            'gemini-1.5-pro-002': '🏆 Gemini 1.5 Pro - Production Ready',
            'gemini-1.5-flash-002': '⚡ Gemini 1.5 Flash - Fast Analysis'
        }
        
        selected_model = st.selectbox(
            "Choose Analysis Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0,
            help="Advanced models provide superior analysis quality and insights"
        )
        
        # Display model information
        analyzer_temp = HedgeFundAnalyzer(api_key, selected_model)
        if selected_model in analyzer_temp.model_configs:
            config = analyzer_temp.model_configs[selected_model]
            st.success(f"✅ **{config['name']}** Selected")
            st.info(f"📋 {config['description']}\n\n💡 **Best for:** {config['best_for']}")
        
        # Enhanced analysis settings
        st.markdown("### ⚙️ Analysis Configuration")
        
        analysis_depth = st.selectbox(
            "📊 Analysis Depth",
            ["Institutional Grade", "Investment Committee Ready", "Deep Due Diligence"],
            index=0,
            help="Higher depth provides more comprehensive analysis"
        )
        
        st.markdown("### 🔧 Advanced Features")
        enable_advanced_caching = st.checkbox(
            "🚀 Advanced Context Caching", 
            value=True,
            help="Reduces token usage and improves analysis consistency"
        )
        
        include_performance_analysis = st.checkbox(
            "📈 Performance Data Analysis", 
            value=True,
            help="Enhanced analysis of uploaded performance data"
        )
        
        include_risk_attribution = st.checkbox(
            "🎯 Risk Factor Attribution", 
            value=True,
            help="Detailed risk decomposition and attribution"
        )
    
    # Initialize analyzer with selected model
    analyzer = HedgeFundAnalyzer(api_key, selected_model)
    
    # Enhanced file upload section
    st.markdown("### 📄 Document Upload Center")
    
    col_upload1, col_upload2 = st.columns([2, 1])
    
    with col_upload1:
        st.markdown("""
        **📋 Supported Documents:**
        - 📊 **Performance Data:** CSV, Excel files with historical returns
        - 📄 **Marketing Materials:** PDF pitch decks, fact sheets
        - 📝 **Text Documents:** Word docs, text files
        - 📈 **Financial Reports:** Performance reports, risk documents
        """)
        
        uploaded_files = st.file_uploader(
            "Select Files for Analysis",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'xls'],
            help="Upload fund materials, performance data, and related documents"
        )
    
    with col_upload2:
        if uploaded_files:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("📁 Files Uploaded", len(uploaded_files))
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Categorize uploaded files
            pdf_files = [f for f in uploaded_files if f.name.endswith('.pdf')]
            data_files = [f for f in uploaded_files if f.name.endswith(('.csv', '.xlsx', '.xls'))]
            doc_files = [f for f in uploaded_files if f.name.endswith(('.docx', '.txt'))]
            
            if pdf_files:
                st.success(f"📄 {len(pdf_files)} PDF documents")
            if data_files:
                st.success(f"📊 {len(data_files)} data files")
            if doc_files:
                st.success(f"📝 {len(doc_files)} text documents")
    
    # Display uploaded files with better organization
    if uploaded_files:
        with st.expander("📋 Uploaded Files Overview", expanded=False):
            file_data = []
            for file in uploaded_files:
                file_type = "📊 Performance Data" if file.name.endswith(('.csv', '.xlsx', '.xls')) else \
                           "📄 PDF Document" if file.name.endswith('.pdf') else \
                           "📝 Text Document"
                
                file_data.append({
                    "File Name": file.name,
                    "Type": file_type,
                    "Size": f"{file.size / 1024:.1f} KB"
                })
            
            st.table(pd.DataFrame(file_data))
    
    return analyzer, model_options, analysis_depth, include_performance_analysis, include_risk_attribution, uploaded_files

def process_analysis(analyzer, model_options, selected_model, uploaded_files):
    """Handle the analysis processing with progress tracking"""
    
    # Enhanced analysis section
    if uploaded_files:
        st.markdown("### 🔍 Advanced Analysis Engine")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        
        with col_btn2:
            analyze_button = st.button(
                "🚀 Start Comprehensive Analysis", 
                type="primary",
                use_container_width=True,
                help="Begin AI-powered fund analysis"
            )
        
        if analyze_button:
            with st.spinner("🧠 Processing documents with advanced AI..."):
                # Create progress tracking
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: Process files
                    status_text.markdown("📄 **Step 1:** Extracting content from uploaded documents...")
                    progress_bar.progress(20)
                    
                    combined_content, performance_dataframes = analyzer.process_uploaded_files(uploaded_files)
                    
                    if not combined_content.strip():
                        st.error("❌ No content could be extracted from uploaded files")
                        return None, None
                    
                    # Step 2: Performance data analysis
                    if performance_dataframes:
                        status_text.markdown("📊 **Step 2:** Analyzing performance data...")
                        progress_bar.progress(40)
                        
                        # Display performance data preview
                        with st.expander("📈 Performance Data Preview", expanded=False):
                            for i, df in enumerate(performance_dataframes):
                                st.markdown(f"**Dataset {i+1}:**")
                                st.dataframe(df.head(), use_container_width=True)
                    
                    # Step 3: AI Analysis
                    status_text.markdown(f"🧠 **Step 3:** Advanced AI analysis with {model_options[selected_model]}...")
                    progress_bar.progress(70)
                    
                    analysis = analyzer.analyze_fund(combined_content, performance_dataframes)
                    
                    progress_bar.progress(100)
                    
                    if analysis:
                        status_text.markdown("✅ **Analysis Complete!** Generating comprehensive report...")
                        time.sleep(1)
                        progress_container.empty()
                        
                        # Store analysis in session state
                        st.session_state['analysis'] = analysis
                        st.session_state['model_used'] = selected_model
                        st.session_state['performance_data'] = performance_dataframes
                        st.session_state['upload_timestamp'] = datetime.datetime.now()
                        
                        st.success("🎉 **Analysis Successfully Completed!**")
                        return analysis, performance_dataframes
                    else:
                        st.error("❌ Analysis failed. Please try again with a different model.")
                        return None, None
    
    return None, None

def display_analysis_metadata(model_options, model_used, performance_data, upload_time):
    """Display analysis metadata in a clean format"""
    
    st.markdown("---")
    st.markdown("## 📊 Comprehensive Fund Analysis Results")
    
    # Analysis metadata
    col_meta1, col_meta2, col_meta3 = st.columns(3)
    with col_meta1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f"**🧠 AI Model Used**<br>{model_options.get(model_used, model_used)}", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_meta2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f"**📊 Data Files**<br>{len(performance_data)} performance datasets", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_meta3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown(f"**⏰ Generated**<br>{upload_time.strftime('%Y-%m-%d %H:%M')}", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def display_executive_summary(analysis):
    """Display executive summary with enhanced styling"""
    
    st.markdown("### 📋 Executive Summary")
    st.markdown(f"""
    <div class="analysis-section">
        <div style="font-size: 1.1rem; line-height: 1.6; color: #2c3e50;">
            {analysis.executive_summary}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance insights if available
    if hasattr(analysis, 'performance_insights') and analysis.performance_insights:
        st.markdown("### 📈 Performance Analysis Insights")
        st.markdown(f"""
        <div class="analysis-section" style="border-left-color: #28a745;">
            <div style="font-size: 1rem; line-height: 1.6; color: #2c3e50;">
                {analysis.performance_insights}
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_key_insights(analysis):
    """Display key insights in enhanced columns"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Strategic Investment Themes")
        for i, theme in enumerate(analysis.key_themes, 1):
            st.markdown(f"""
            <div style="background-color: #e8f4fd; padding: 0.8rem; margin: 0.5rem 0; border-radius: 8px; border-left: 3px solid #1f77b4;">
                <strong>{i}.</strong> {theme}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### ✅ Investment Advantages")
        for i, pro in enumerate(analysis.pros, 1):
            st.markdown(f"""
            <div style="background-color: #e8f5e8; padding: 0.8rem; margin: 0.5rem 0; border-radius: 8px; border-left: 3px solid #28a745;">
                <strong>{i}.</strong> {pro}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🏆 Competitive Differentiation")
        st.markdown(f"""
        <div class="analysis-section" style="border-left-color: #ffc107;">
            <div style="font-size: 1rem; line-height: 1.6; color: #2c3e50;">
                {analysis.competitive_edge}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚠️ Risk Considerations")
        for i, con in enumerate(analysis.cons, 1):
            st.markdown(f"""
            <div style="background-color: #fdebeb; padding: 0.8rem; margin: 0.5rem 0; border-radius: 8px; border-left: 3px solid #dc3545;">
                <strong>{i}.</strong> {con}
            </div>
            """, unsafe_allow_html=True)

def display_performance_analytics(include_performance_analysis, performance_data):
    """Display performance visualizations and analytics"""
    
    if include_performance_analysis and performance_data:
        st.markdown("### 📊 Performance Analytics Dashboard")
        
        tab1, tab2, tab3 = st.tabs(["📈 Performance Chart", "🎯 Risk Metrics", "📋 Data Summary"])
        
        with tab1:
            perf_chart = create_performance_visualization(performance_data)
            st.plotly_chart(perf_chart, use_container_width=True)
        
        with tab2:
            risk_chart = create_risk_metrics_chart(performance_data)
            st.plotly_chart(risk_chart, use_container_width=True)
        
        with tab3:
            st.markdown("**📊 Performance Data Summary:**")
            for i, df in enumerate(performance_data):
                with st.expander(f"Dataset {i+1} - {df.shape[0]} rows, {df.shape[1]} columns"):
                    col_summary1, col_summary2 = st.columns(2)
                    with col_summary1:
                        st.markdown("**Data Overview:**")
                        st.dataframe(df.describe(), use_container_width=True)
                    with col_summary2:
                        st.markdown("**Sample Data:**")
                        st.dataframe(df.head(), use_container_width=True)

def display_team_analysis(analysis):
    """Display enhanced team analysis"""
    
    if analysis.key_people:
        st.markdown("### 👥 Leadership Team Analysis")
        for person in analysis.key_people:
            with st.expander(f"🔍 {person['name']} - {person['role']}", expanded=False):
                col_bg, col_risk = st.columns([2, 1])
                with col_bg:
                    st.markdown("**Background & Track Record:**")
                    st.write(person['background'])
                with col_risk:
                    if 'risk_assessment' in person:
                        st.markdown("**Risk Assessment:**")
                        st.write(person['risk_assessment'])

def display_risk_return_analysis(analysis):
    """Display enhanced risk and return analysis"""
    
    st.markdown("### 🔬 Advanced Risk & Return Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 🎲 Multi-Dimensional Risk Assessment")
        st.markdown(f"""
        <div class="analysis-section" style="border-left-color: #ff7f0e;">
            <div style="font-size: 1rem; line-height: 1.6; color: #2c3e50;">
                {analysis.risk_assessment}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("#### 📈 Quantitative Return Analysis")
        st.markdown(f"""
        <div class="analysis-section" style="border-left-color: #17a2b8;">
            <div style="font-size: 1rem; line-height: 1.6; color: #2c3e50;">
                {analysis.return_analysis}
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_benchmark_comparison(analysis):
    """Display benchmark comparison if available"""
    
    if hasattr(analysis, 'benchmark_comparison') and analysis.benchmark_comparison:
        st.markdown("### 📊 Benchmark & Peer Comparison")
        st.markdown(f"""
        <div class="analysis-section" style="border-left-color: #6f42c1;">
            <div style="font-size: 1rem; line-height: 1.6; color: #2c3e50;">
                {analysis.benchmark_comparison}
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_investment_recommendation(analysis):
    """Display enhanced investment recommendation with conviction scoring"""
    
    st.markdown("### 💡 Investment Committee Recommendation")
    
    recommendation_text = analysis.recommendation
    conviction_match = re.search(r'conviction.*?(\d+)', recommendation_text.lower())
    conviction_level = int(conviction_match.group(1)) if conviction_match else None
    
    # Determine recommendation styling
    if "strong buy" in recommendation_text.lower() or "highly recommend" in recommendation_text.lower():
        rec_color = "#d4edda"
        rec_border = "#28a745"
        rec_icon = "🟢"
        rec_badge = "STRONG BUY"
    elif "buy" in recommendation_text.lower() or "recommend" in recommendation_text.lower():
        rec_color = "#d1ecf1" 
        rec_border = "#17a2b8"
        rec_icon = "🔵"
        rec_badge = "BUY"
    elif "hold" in recommendation_text.lower() or "neutral" in recommendation_text.lower():
        rec_color = "#fff3cd"
        rec_border = "#ffc107"
        rec_icon = "🟡"
        rec_badge = "HOLD"
    else:
        rec_color = "#f8d7da"
        rec_border = "#dc3545"
        rec_icon = "🔴"
        rec_badge = "AVOID"
    
    st.markdown(f"""
    <div class="recommendation-box" style='background-color: {rec_color}; border-left: 8px solid {rec_border};'>
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 2rem; margin-right: 1rem;">{rec_icon}</span>
            <div>
                <h3 style="margin: 0; color: #2c3e50;">Investment Recommendation: <span class="highlight-text">{rec_badge}</span></h3>
                {f'<p style="margin: 0.5rem 0; font-weight: 600; font-size: 1.1rem;">Conviction Level: {conviction_level}/10</p>' if conviction_level else ''}
            </div>
        </div>
        <div style="font-size: 1.1rem; line-height: 1.6; color: #2c3e50;">
            {recommendation_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_export_data(analysis, model_used, analysis_depth, performance_data, model_options):
    """Create comprehensive export data structure"""
    
    analysis_dict = {
        'analysis_metadata': {
            'model_used': model_used,
            'analysis_depth': analysis_depth,
            'generated_at': datetime.datetime.now().isoformat(),
            'performance_datasets': len(performance_data),
            'advanced_features': ['context_caching', 'performance_analysis', 'enhanced_prompting']
        },
        'executive_summary': analysis.executive_summary,
        'key_themes': analysis.key_themes,
        'competitive_edge': analysis.competitive_edge,
        'key_people': analysis.key_people,
        'pros': analysis.pros,
        'cons': analysis.cons,
        'risk_assessment': analysis.risk_assessment,
        'return_analysis': analysis.return_analysis,
        'recommendation': analysis.recommendation
    }
    
    if hasattr(analysis, 'performance_insights'):
        analysis_dict['performance_insights'] = analysis.performance_insights
    if hasattr(analysis, 'benchmark_comparison'):
        analysis_dict['benchmark_comparison'] = analysis.benchmark_comparison
    
    return analysis_dict

def create_text_report(analysis, model_used, analysis_depth, performance_data, model_options):
    """Create comprehensive text report"""
    
    text_summary = f"""
PROFESSIONAL HEDGE FUND ANALYSIS REPORT
═══════════════════════════════════════
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
AI Model: {model_options.get(model_used, model_used)}
Analysis Depth: {analysis_depth}
Performance Datasets: {len(performance_data)}

EXECUTIVE SUMMARY
─────────────────
{analysis.executive_summary}

STRATEGIC INVESTMENT THEMES
───────────────────────────
{chr(10).join([f"{i}. {theme}" for i, theme in enumerate(analysis.key_themes, 1)])}

COMPETITIVE DIFFERENTIATION
──────────────────────────
{analysis.competitive_edge}

INVESTMENT ADVANTAGES
────────────────────
{chr(10).join([f"{i}. {pro}" for i, pro in enumerate(analysis.pros, 1)])}

RISK CONSIDERATIONS
──────────────────
{chr(10).join([f"{i}. {con}" for i, con in enumerate(analysis.cons, 1)])}

MULTI-DIMENSIONAL RISK ASSESSMENT
─────────────────────────────────
{analysis.risk_assessment}

QUANTITATIVE RETURN ANALYSIS
───────────────────────────
{analysis.return_analysis}

INVESTMENT COMMITTEE RECOMMENDATION
──────────────────────────────────
{analysis.recommendation}

{'PERFORMANCE INSIGHTS' + chr(10) + '─' * 20 + chr(10) + analysis.performance_insights if hasattr(analysis, 'performance_insights') and analysis.performance_insights else ''}

{'BENCHMARK COMPARISON' + chr(10) + '─' * 18 + chr(10) + analysis.benchmark_comparison if hasattr(analysis, 'benchmark_comparison') and analysis.benchmark_comparison else ''}

═══════════════════════════════════════
Analysis powered by: {model_options.get(model_used, model_used)}
Generated by Professional Fund Analysis Platform
    """
    
    return text_summary

def create_ic_summary(analysis, model_used, model_options):
    """Create Investment Committee summary"""
    
    recommendation_text = analysis.recommendation
    conviction_match = re.search(r'conviction.*?(\d+)', recommendation_text.lower())
    conviction_level = int(conviction_match.group(1)) if conviction_match else None
    
    # Determine recommendation badge
    if "strong buy" in recommendation_text.lower() or "highly recommend" in recommendation_text.lower():
        rec_badge = "STRONG BUY"
    elif "buy" in recommendation_text.lower() or "recommend" in recommendation_text.lower():
        rec_badge = "BUY"
    elif "hold" in recommendation_text.lower() or "neutral" in recommendation_text.lower():
        rec_badge = "HOLD"
    else:
        rec_badge = "AVOID"
    
    ic_summary = f"""
INVESTMENT COMMITTEE MEMORANDUM
═════════════════════════════

Fund Analysis Summary
Date: {datetime.datetime.now().strftime('%B %d, %Y')}
Senior Analyst: AI-Powered Analysis
Model: {model_options.get(model_used, 'Advanced AI')}

RECOMMENDATION: {rec_badge}
{f'CONVICTION: {conviction_level}/10' if conviction_level else ''}

KEY HIGHLIGHTS:
• Strategy: {analysis.key_themes[0] if analysis.key_themes else 'N/A'}
• Advantage: {analysis.pros[0][:100] + '...' if analysis.pros else 'N/A'}
• Risk: {analysis.cons[0][:100] + '...' if analysis.cons else 'N/A'}

NEXT STEPS:
□ Investment Committee Review
□ Management Presentation
□ Reference Calls & Due Diligence
□ Final Allocation Decision

ANALYSIS FEATURES:
• Performance Data Analysis
• Advanced AI Reasoning
• Multi-Dimensional Risk Assessment
• Competitive Positioning Analysis

Full detailed analysis available in comprehensive report.

═════════════════════════════
Professional Fund Analysis Platform
    """
    
    return ic_summary

def display_export_section(analysis, model_used, analysis_depth, performance_data, model_options):
    """Display enhanced export functionality"""
    
    st.markdown("### 📤 Export Analysis Reports")
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        # Enhanced JSON export
        analysis_dict = create_export_data(analysis, model_used, analysis_depth, performance_data, model_options)
        json_str = json.dumps(analysis_dict, indent=2)
        
        st.download_button(
            label="📋 Download JSON Report",
            data=json_str,
            file_name=f"fund_analysis_{model_used}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col6:
        # Enhanced text summary
        text_summary = create_text_report(analysis, model_used, analysis_depth, performance_data, model_options)
        
        st.download_button(
            label="📄 Download Text Report",
            data=text_summary,
            file_name=f"fund_analysis_{model_used}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col7:
        # Investment Committee Summary
        ic_summary = create_ic_summary(analysis, model_used, model_options)
        
        st.download_button(
            label="📊 Download IC Summary",
            data=ic_summary,
            file_name=f"IC_memo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col8:
        # Performance data export
        if performance_data:
            # Combine all performance data
            combined_perf_data = []
            for i, df in enumerate(performance_data):
                df_copy = df.copy()
                df_copy['Source_Dataset'] = f'Dataset_{i+1}'
                combined_perf_data.append(df_copy)
            
            if combined_perf_data:
                combined_df = pd.concat(combined_perf_data, ignore_index=True)
                csv_data = combined_df.to_csv(index=False)
                
                st.download_button(
                    label="📈 Download Performance Data",
                    data=csv_data,
                    file_name=f"performance_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("No performance data to export")

def main_setup():
    """Setup function from Part 6 - returns configuration variables"""
    
    # Enhanced header with better styling
    st.markdown('<h1 class="main-header">🏦 Professional Fund Analysis Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Institutional-Grade Investment Analysis • Powered by Advanced AI</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### 🚀 Configuration Panel")
        
        # API Key input
        api_key = st.text_input(
            "🔑 Gemini API Key", 
            type="password",
            help="Enter your Google Gemini API key for advanced analysis",
            placeholder="Enter API key here..."
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your Gemini API key to begin analysis")
            st.info("💡 Get your API key from Google AI Studio")
            return None
        
        # Model selection with clearer descriptions
        st.markdown("### 🧠 AI Model Selection")
        
        model_options = {
            'gemini-2.0-flash-exp': '🌟 Gemini 2.0 Flash (Latest) - Most Advanced',
            'gemini-exp-1121': '🔬 Gemini Experimental - Enhanced Reasoning',
            'gemini-1.5-pro-002': '🏆 Gemini 1.5 Pro - Production Ready',
            'gemini-1.5-flash-002': '⚡ Gemini 1.5 Flash - Fast Analysis'
        }
        
        selected_model = st.selectbox(
            "Choose Analysis Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0,
            help="Advanced models provide superior analysis quality and insights"
        )
        
        # Display model information
        analyzer_temp = HedgeFundAnalyzer(api_key, selected_model)
        if selected_model in analyzer_temp.model_configs:
            config = analyzer_temp.model_configs[selected_model]
            st.success(f"✅ **{config['name']}** Selected")
            st.info(f"📋 {config['description']}\n\n💡 **Best for:** {config['best_for']}")
        
        # Enhanced analysis settings
        st.markdown("### ⚙️ Analysis Configuration")
        
        analysis_depth = st.selectbox(
            "📊 Analysis Depth",
            ["Institutional Grade", "Investment Committee Ready", "Deep Due Diligence"],
            index=0,
            help="Higher depth provides more comprehensive analysis"
        )
        
        st.markdown("### 🔧 Advanced Features")
        enable_advanced_caching = st.checkbox(
            "🚀 Advanced Context Caching", 
            value=True,
            help="Reduces token usage and improves analysis consistency"
        )
        
        include_performance_analysis = st.checkbox(
            "📈 Performance Data Analysis", 
            value=True,
            help="Enhanced analysis of uploaded performance data"
        )
        
        include_risk_attribution = st.checkbox(
            "🎯 Risk Factor Attribution", 
            value=True,
            help="Detailed risk decomposition and attribution"
        )
    
    # Initialize analyzer with selected model
    analyzer = HedgeFundAnalyzer(api_key, selected_model)
    
    # Enhanced file upload section
    st.markdown("### 📄 Document Upload Center")
    
    col_upload1, col_upload2 = st.columns([2, 1])
    
    with col_upload1:
        st.markdown("""
        **📋 Supported Documents:**
        - 📊 **Performance Data:** CSV, Excel files with historical returns
        - 📄 **Marketing Materials:** PDF pitch decks, fact sheets
        - 📝 **Text Documents:** Word docs, text files
        - 📈 **Financial Reports:** Performance reports, risk documents
        """)
        
        uploaded_files = st.file_uploader(
            "Select Files for Analysis",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'xls'],
            help="Upload fund materials, performance data, and related documents"
        )
    
    with col_upload2:
        if uploaded_files:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("📁 Files Uploaded", len(uploaded_files))
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Categorize uploaded files
            pdf_files = [f for f in uploaded_files if f.name.endswith('.pdf')]
            data_files = [f for f in uploaded_files if f.name.endswith(('.csv', '.xlsx', '.xls'))]
            doc_files = [f for f in uploaded_files if f.name.endswith(('.docx', '.txt'))]
            
            if pdf_files:
                st.success(f"📄 {len(pdf_files)} PDF documents")
            if data_files:
                st.success(f"📊 {len(data_files)} data files")
            if doc_files:
                st.success(f"📝 {len(doc_files)} text documents")
    
    # Display uploaded files with better organization
    if uploaded_files:
        with st.expander("📋 Uploaded Files Overview", expanded=False):
            file_data = []
            for file in uploaded_files:
                file_type = "📊 Performance Data" if file.name.endswith(('.csv', '.xlsx', '.xls')) else \
                           "📄 PDF Document" if file.name.endswith('.pdf') else \
                           "📝 Text Document"
                
                file_data.append({
                    "File Name": file.name,
                    "Type": file_type,
                    "Size": f"{file.size / 1024:.1f} KB"
                })
            
            st.table(pd.DataFrame(file_data))
    
    return analyzer, model_options, analysis_depth, include_performance_analysis, include_risk_attribution, uploaded_files

def main():
    """Complete main function combining all components"""
    
    # Setup and configuration
    result = main_setup()
    if result is None:
        return
    
    analyzer, model_options, analysis_depth, include_performance_analysis, include_risk_attribution, uploaded_files = result
    
    # Get selected model from analyzer
    selected_model = analyzer.model_name
    
    # Process analysis
    analysis, performance_data = process_analysis(analyzer, model_options, selected_model, uploaded_files)
    
    # Display results if analysis exists in session state
    if 'analysis' in st.session_state:
        analysis = st.session_state['analysis']
        model_used = st.session_state.get('model_used', 'unknown')
        performance_data = st.session_state.get('performance_data', [])
        upload_time = st.session_state.get('upload_timestamp', datetime.datetime.now())
        
        # Display analysis metadata
        display_analysis_metadata(model_options, model_used, performance_data, upload_time)
        
        # Display executive summary
        display_executive_summary(analysis)
        
        # Display key insights
        display_key_insights(analysis)
        
        # Display performance analytics
        display_performance_analytics(include_performance_analysis, performance_data)
        
        # Display team analysis
        display_team_analysis(analysis)
        
        # Display risk and return analysis
        display_risk_return_analysis(analysis)
        
        # Display benchmark comparison
        display_benchmark_comparison(analysis)
        
        # Display investment recommendation
        display_investment_recommendation(analysis)
        
        # Display export section
        display_export_section(analysis, model_used, analysis_depth, performance_data, model_options)

if __name__ == "__main__":
    main()
