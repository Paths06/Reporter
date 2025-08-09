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
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
import re

# Configure page
st.set_page_config(
    page_title="Hedge Fund Investment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class FundAnalysis:
    """Data class for fund analysis results"""
    executive_summary: str
    key_themes: List[str]
    competitive_edge: str
    key_people: List[Dict[str, str]]
    pros: List[str]
    cons: List[str]
    risk_assessment: str
    return_analysis: str
    recommendation: str

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
    
    def process_uploaded_files(self, uploaded_files) -> str:
        """Process and combine text from all uploaded files"""
        combined_text = ""
        
        for file in uploaded_files:
            file_extension = file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                text = self.extract_text_from_pdf(file)
            elif file_extension == 'docx':
                text = self.extract_text_from_docx(file)
            elif file_extension == 'txt':
                text = str(file.read(), "utf-8")
            else:
                st.warning(f"Unsupported file type: {file.name}")
                continue
                
            combined_text += f"\n\n--- Content from {file.name} ---\n\n{text}"
        
        return combined_text
    
    def analyze_fund(self, content: str) -> FundAnalysis:
        """Analyze fund using advanced Gemini AI models with enhanced prompting"""
        
        # Create context cache
        cache_name = self.create_context_cache(content, "advanced_fund_analysis")
        
        # Enhanced analysis prompt for advanced models
        analysis_prompt = """
        As a senior investment partner with 20+ years of hedge fund experience and deep quantitative expertise, 
        perform an institutional-grade analysis of the provided fund materials. Use advanced reasoning and 
        pattern recognition to identify subtle risks and opportunities that junior analysts might miss.
        
        Leverage your training on financial markets, alternative investments, and risk management to provide 
        sophisticated insights. Consider market cycles, regime changes, and systemic risks.
        
        Return analysis in this precise JSON format:
        
        {
            "executive_summary": "Sophisticated 4-5 paragraph executive summary with nuanced investment thesis, addressing strategy sophistication, execution capability, market positioning, and institutional fit. Include quantitative insights where available.",
            
            "key_themes": [
                "Advanced theme 1 with market context",
                "Strategic theme 2 with competitive analysis", 
                "Risk theme 3 with regime awareness",
                "Opportunity theme 4 with timing considerations",
                "Operational theme 5 with scalability assessment"
            ],
            
            "competitive_edge": "Detailed analysis of sustainable competitive advantages, moats, barriers to entry, and differentiation sustainability. Assess whether edge is structural, informational, or execution-based.",
            
            "key_people": [
                {
                    "name": "Full Name",
                    "role": "Specific Title", 
                    "background": "Detailed background with track record, previous funds, educational pedigree, and key achievements",
                    "risk_assessment": "Personal key-man risk and succession planning"
                }
            ],
            
            "pros": [
                "Sophisticated pro 1: Detailed advantage with market context and sustainability analysis",
                "Strategic pro 2: Competitive positioning with barriers to entry assessment", 
                "Execution pro 3: Operational excellence with scalability considerations",
                "Team pro 4: Human capital advantages with retention and succession planning",
                "Timing pro 5: Market opportunity with cycle positioning and catalyst analysis",
                "Risk-adjusted pro 6: Downside protection and risk management sophistication"
            ],
            
            "cons": [
                "Material con 1: Significant risk with impact analysis and mitigation assessment",
                "Strategic con 2: Competitive vulnerability with market evolution risks",
                "Operational con 3: Scalability or capacity constraints with growth limitations", 
                "Market con 4: Cycle sensitivity or regime change risks with stress scenarios",
                "Liquidity con 5: Redemption risks or illiquidity concerns with portfolio impact"
            ],
            
            "risk_assessment": "Comprehensive multi-dimensional risk analysis covering: Market Risk (beta, factor exposures, regime sensitivity), Credit Risk (counterparty, liquidity), Operational Risk (key-man, systems, compliance), Structural Risk (leverage, liquidity terms), and Systemic Risk (correlation, tail events). Include stress testing insights and scenario analysis.",
            
            "return_analysis": "Sophisticated return analysis including: risk-adjusted metrics (Sharpe, Sortino, Calmar ratios), drawdown analysis, return distribution characteristics, factor attribution, benchmark comparisons, capacity constraints impact on returns, and forward-looking return expectations with confidence intervals.",
            
            "recommendation": "Institutional investment committee-ready recommendation with specific allocation sizing, timing considerations, portfolio fit analysis, risk budget impact, and implementation pathway. Include conviction level (1-10) and key monitoring metrics."
        }
        
        Advanced Analysis Framework:
        
        **Strategy Deep Dive:**
        - Alpha source sustainability and decay risks
        - Strategy capacity and scalability limits  
        - Market regime sensitivity and adaptability
        - Innovation and strategy evolution capability
        
        **Risk Management Excellence:**
        - Portfolio construction sophistication
        - Risk budgeting and attribution systems
        - Stress testing and scenario planning
        - Liquidity management and portfolio liquidity
        
        **Operational Infrastructure:**
        - Technology and data advantages
        - Research process and investment committee structure
        - Compliance and regulatory preparedness
        - Business continuity and succession planning
        
        **Market Positioning:**
        - Competitive landscape evolution
        - Regulatory environment impact
        - Market structure changes and adaptation
        - Client base quality and stickiness
        
        **Quantitative Assessment:**
        - Statistical significance of track record
        - Return attribution and factor exposures
        - Correlation analysis and portfolio fit
        - Capacity utilization and performance impact
        
        Use sophisticated financial reasoning. Identify non-obvious risks and opportunities. 
        Provide actionable insights for institutional allocators.
        """
        
        try:
            # Enhanced generation config for advanced models
            generation_config = {
                "temperature": 0.1,  # Lower for more consistent analysis
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 8192,  # Increased for comprehensive analysis
                "response_mime_type": "application/json"
            }
            
            if cache_name:
                # Use cached context with advanced model
                model_with_cache = genai.GenerativeModel(
                    self.model_name, 
                    cached_content=cache_name,
                    generation_config=generation_config
                )
                response = model_with_cache.generate_content(analysis_prompt)
            else:
                # Fallback without cache
                model_with_config = genai.GenerativeModel(
                    self.model_name,
                    generation_config=generation_config
                )
                full_prompt = f"{analysis_prompt}\n\nFund Materials:\n{content}"
                response = model_with_config.generate_content(full_prompt)
            
            # Enhanced JSON parsing with error handling
            response_text = response.text.strip()
            
            # Clean up response format
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            # Try to parse JSON
            try:
                analysis_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from response if wrapped in text
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")
            
            return FundAnalysis(**analysis_data)
            
        except json.JSONDecodeError as e:
            st.error(f"Advanced model JSON parsing failed: {e}")
            st.error(f"Response preview: {response_text[:500]}...")
            return None
        except Exception as e:
            st.error(f"Advanced analysis failed: {e}")
            return None

def create_risk_return_chart(analysis: FundAnalysis):
    """Create risk-return visualization"""
    # This is a placeholder - in production, you'd extract actual metrics
    fig = go.Figure()
    
    # Sample data points for visualization
    fig.add_trace(go.Scatter(
        x=[15, 20, 12, 18, 25],
        y=[8, 12, 6, 10, 15],
        mode='markers+text',
        marker=dict(size=12, color=['red', 'green', 'blue', 'orange', 'purple']),
        text=['Fund', 'S&P 500', 'Bonds', 'Benchmark', 'Peers'],
        textposition='top center',
        name='Risk-Return Profile'
    ))
    
    fig.update_layout(
        title='Risk-Return Analysis',
        xaxis_title='Risk (Volatility %)',
        yaxis_title='Return (%)',
        height=400
    )
    
    return fig

def main():
    st.title("🏦 Advanced Hedge Fund Investment Analysis Dashboard")
    st.markdown("*Senior Partner Investment Analysis Platform - Powered by Advanced AI*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🚀 Advanced Model Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Gemini API Key", 
            type="password",
            help="Enter your Google Gemini API key for advanced models"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your Gemini API key to continue")
            return
        
        # Model selection
        st.subheader("🧠 AI Model Selection")
        
        model_options = {
            'gemini-2.0-flash-exp': '🌟 Gemini 2.0 Flash (Experimental) - Latest & Greatest',
            'gemini-exp-1121': '🔬 Gemini Experimental 1121 - Advanced Reasoning',
            'gemini-1.5-pro-002': '🏆 Gemini 1.5 Pro (Latest) - Production Ready',
            'gemini-1.5-flash-002': '⚡ Gemini 1.5 Flash (Latest) - Fast & Efficient'
        }
        
        selected_model = st.selectbox(
            "Choose Analysis Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0,
            help="Advanced models provide superior analysis quality"
        )
        
        # Display model info
        analyzer_temp = HedgeFundAnalyzer(api_key, selected_model)
        if selected_model in analyzer_temp.model_configs:
            config = analyzer_temp.model_configs[selected_model]
            st.info(f"**{config['name']}**\n\n{config['description']}\n\n*Best for: {config['best_for']}*")
        
        st.success(f"✅ {model_options[selected_model]} configured")
        
        # Advanced analysis settings
        st.header("⚙️ Analysis Settings")
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Institutional Grade", "Investment Committee Ready", "Deep Due Diligence"],
            index=0,
            help="Higher depth = more comprehensive analysis"
        )
        
        enable_advanced_caching = st.checkbox(
            "Advanced Context Caching", 
            value=True,
            help="Reduces token usage and improves consistency"
        )
        
        include_quantitative_analysis = st.checkbox("Quantitative Analysis", value=True)
        include_risk_attribution = st.checkbox("Risk Factor Attribution", value=True)
        include_scenario_analysis = st.checkbox("Scenario Analysis", value=True)
    
    # Initialize analyzer with selected model
    analyzer = HedgeFundAnalyzer(api_key, selected_model)
    
    # File upload section
    st.header("📄 Document Upload")
    st.markdown("Upload fund marketing materials, pitch decks, performance reports, and related documents")
    
    uploaded_files = st.file_uploader(
        "Select Files",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt'],
        help="Supported formats: PDF, Word documents, Text files"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded")
        
        # Display uploaded files
        with st.expander("📋 Uploaded Files"):
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size} bytes)")
    
    # Analysis section
    if uploaded_files and st.button("🔍 Advanced AI Analysis", type="primary"):
        with st.spinner(f"Processing documents with {model_options[selected_model]}..."):
            # Process files
            combined_content = analyzer.process_uploaded_files(uploaded_files)
            
            if not combined_content.strip():
                st.error("❌ No content could be extracted from uploaded files")
                return
            
            # Show analysis progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🧠 Advanced AI model analyzing fund materials...")
            progress_bar.progress(25)
            
            # Analyze fund with advanced model
            analysis = analyzer.analyze_fund(combined_content)
            progress_bar.progress(100)
            
            if analysis:
                status_text.text("✅ Advanced analysis completed successfully!")
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()
                
                # Store analysis in session state
                st.session_state['analysis'] = analysis
                st.session_state['model_used'] = selected_model
            else:
                st.error("❌ Advanced analysis failed. Please try again with different model.")
                return
    
    # Display results if analysis exists
    if 'analysis' in st.session_state:
        analysis = st.session_state['analysis']
        model_used = st.session_state.get('model_used', 'unknown')
        
        st.header("📊 Advanced Fund Analysis Results")
        
        # Show model used
        st.info(f"🧠 **Analysis powered by:** {model_options.get(model_used, model_used)}")
        
        # Executive Summary with enhanced formatting
        st.subheader("📋 Executive Summary")
        st.markdown(f"<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;'>{analysis.executive_summary}</div>", unsafe_allow_html=True)
        
        # Key metrics in columns with enhanced styling
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Strategic Investment Themes")
            for i, theme in enumerate(analysis.key_themes, 1):
                st.markdown(f"**{i}.** {theme}")
            
            st.subheader("✅ Investment Advantages")
            for i, pro in enumerate(analysis.pros, 1):
                st.markdown(f"**{i}.** {pro}")
        
        with col2:
            st.subheader("🏆 Competitive Differentiation")
            st.markdown(f"<div style='background-color: #e8f5e8; padding: 15px; border-radius: 8px;'>{analysis.competitive_edge}</div>", unsafe_allow_html=True)
            
            st.subheader("⚠️ Risk Considerations")
            for i, con in enumerate(analysis.cons, 1):
                st.markdown(f"**{i}.** {con}")
        
        # Enhanced Key People section
        st.subheader("👥 Leadership Team Analysis")
        if analysis.key_people:
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
        
        # Enhanced Risk and Return Analysis
        st.subheader("🔬 Advanced Risk & Return Analysis")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### 🎲 Multi-Dimensional Risk Assessment")
            st.markdown(f"<div style='background-color: #fff2e6; padding: 15px; border-radius: 8px; border-left: 4px solid #ff7f0e;'>{analysis.risk_assessment}</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("### 📈 Quantitative Return Analysis")
            st.markdown(f"<div style='background-color: #e6f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8;'>{analysis.return_analysis}</div>", unsafe_allow_html=True)
        
        # Enhanced Investment Recommendation with conviction scoring
        st.subheader("💡 Investment Committee Recommendation")
        
        # Extract conviction level if present
        recommendation_text = analysis.recommendation
        conviction_match = re.search(r'conviction.*?(\d+)', recommendation_text.lower())
        conviction_level = int(conviction_match.group(1)) if conviction_match else None
        
        # Create recommendation container with styling
        if "strong buy" in recommendation_text.lower() or "highly recommend" in recommendation_text.lower():
            rec_color = "#d4edda"
            rec_border = "#28a745"
            rec_icon = "🟢"
        elif "buy" in recommendation_text.lower() or "recommend" in recommendation_text.lower():
            rec_color = "#d1ecf1" 
            rec_border = "#17a2b8"
            rec_icon = "🔵"
        elif "hold" in recommendation_text.lower() or "neutral" in recommendation_text.lower():
            rec_color = "#fff3cd"
            rec_border = "#ffc107"
            rec_icon = "🟡"
        else:
            rec_color = "#f8d7da"
            rec_border = "#dc3545"
            rec_icon = "🔴"
        
        st.markdown(f"""
        <div style='background-color: {rec_color}; padding: 20px; border-radius: 10px; border-left: 5px solid {rec_border}; margin: 10px 0;'>
            <h4>{rec_icon} Investment Recommendation</h4>
            <p>{recommendation_text}</p>
            {f'<p><strong>Conviction Level: {conviction_level}/10</strong></p>' if conviction_level else ''}
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced Risk-Return Chart with advanced features
        if include_quantitative_analysis:
            st.subheader("📊 Advanced Portfolio Analytics")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Risk-Return Profile", "Risk Attribution", "Scenario Analysis"])
            
            with tab1:
                risk_chart = create_risk_return_chart(analysis)
                st.plotly_chart(risk_chart, use_container_width=True)
            
            with tab2:
                st.info("🔧 Risk attribution analysis would be displayed here with actual performance data")
                
            with tab3:
                st.info("🔧 Monte Carlo scenarios and stress testing results would be displayed here")
        
        # Model comparison insights
        if model_used in ['gemini-2.0-flash-exp', 'gemini-exp-1121']:
            st.subheader("🌟 Advanced AI Insights")
            st.success(f"""
            **Enhanced Analysis Features Utilized:**
            - 🧠 Advanced reasoning and pattern recognition
            - 📊 Sophisticated quantitative analysis capabilities  
            - 🔍 Nuanced risk identification and assessment
            - 📈 Enhanced market context and regime awareness
            - ⚡ Superior context understanding and synthesis
            """)
        
        # Enhanced Export functionality with model attribution
        st.header("📤 Export Advanced Analysis")
        
        col5, col6, col7 = st.columns(3)         
        with col5:
            # Enhanced JSON export with model metadata
            analysis_dict = {
                'analysis_metadata': {
                    'model_used': model_used,
                    'analysis_depth': analysis_depth,
                    'generated_at': datetime.datetime.now().isoformat(),
                    'advanced_features': ['context_caching', 'enhanced_prompting', 'quantitative_analysis']
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
            
            json_str = json.dumps(analysis_dict, indent=2)
            st.download_button(
                label="📋 Download Advanced JSON Report",
                data=json_str,
                file_name=f"advanced_fund_analysis_{model_used}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col6:
            # Enhanced text summary with model attribution
            text_summary = f"""
ADVANCED HEDGE FUND INVESTMENT ANALYSIS REPORT
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
AI Model: {model_options.get(model_used, model_used)}
Analysis Depth: {analysis_depth}

EXECUTIVE SUMMARY
{analysis.executive_summary}

STRATEGIC INVESTMENT THEMES
{chr(10).join([f"{i}. {theme}" for i, theme in enumerate(analysis.key_themes, 1)])}

COMPETITIVE DIFFERENTIATION
{analysis.competitive_edge}

INVESTMENT ADVANTAGES
{chr(10).join([f"{i}. {pro}" for i, pro in enumerate(analysis.pros, 1)])}

RISK CONSIDERATIONS
{chr(10).join([f"{i}. {con}" for i, con in enumerate(analysis.cons, 1)])}

MULTI-DIMENSIONAL RISK ASSESSMENT
{analysis.risk_assessment}

QUANTITATIVE RETURN ANALYSIS
{analysis.return_analysis}

INVESTMENT COMMITTEE RECOMMENDATION
{analysis.recommendation}

---
Analysis powered by Advanced AI: {model_options.get(model_used, model_used)}
            """
            
            st.download_button(
                label="📄 Download Enhanced Text Report",
                data=text_summary,
                file_name=f"advanced_fund_analysis_{model_used}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        with col7:
            # Investment Committee PDF-ready summary
            pdf_summary = f"""
INVESTMENT COMMITTEE MEMORANDUM

Fund Analysis Summary
Date: {datetime.datetime.now().strftime('%B %d, %Y')}
Analyst: Senior Investment Partner
AI Model: {model_options.get(model_used, 'Advanced AI')}

RECOMMENDATION: {analysis.recommendation[:100]}...

KEY HIGHLIGHTS:
• {analysis.key_themes[0] if analysis.key_themes else 'N/A'}
• {analysis.pros[0] if analysis.pros else 'N/A'}
• {analysis.cons[0] if analysis.cons else 'N/A'}

NEXT STEPS:
- Investment Committee Review
- Due Diligence Deep Dive
- Reference Calls
- Final Allocation Decision

Full analysis available in detailed report.
            """
            
            st.download_button(
                label="📊 Download IC Summary",
                data=pdf_summary,
                file_name=f"IC_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
