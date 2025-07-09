import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """Handles generation of visualizations from data and natural language requests."""
    
    def __init__(self):
        self.supported_charts = {
            'bar': self._create_bar_chart,
            'line': self._create_line_chart,
            'scatter': self._create_scatter_plot,
            'pie': self._create_pie_chart,
            'histogram': self._create_histogram,
            'box': self._create_box_plot,
        }
    
    def detect_chart_type(self, user_input: str) -> str:
        """Detect the type of chart requested by the user.
        
        Args:
            user_input: The user's input text
            
        Returns:
            str: The detected chart type (defaults to 'bar' if not detected)
        """
        user_input = user_input.lower()
        
        # Check for specific chart type mentions
        for chart_type in self.supported_charts.keys():
            if chart_type in user_input:
                return chart_type
        
        # Check for common phrases that indicate chart types
        if 'compare' in user_input or 'show me the difference' in user_input:
            return 'bar'
        elif 'over time' in user_input or 'trend' in user_input:
            return 'line'
        elif 'distribution' in user_input or 'frequency' in user_input:
            return 'histogram'
        elif 'relationship' in user_input or 'correlation' in user_input:
            return 'scatter'
        elif 'percentage' in user_input or 'proportion' in user_input:
            return 'pie'
        
        # Default to bar chart
        return 'bar'
    
    def _extract_columns(self, df: pd.DataFrame, user_input: str) -> Tuple[str, str]:
        """Extract column names from user input or use defaults.
        
        Args:
            df: The DataFrame to extract columns from
            user_input: The user's input text
            
        Returns:
            Tuple of (x_column, y_column) column names
        """
        # Look for column names in the user input
        columns_found = []
        for col in df.columns:
            if col.lower() in user_input.lower():
                columns_found.append(col)
        
        # If we found at least one column, use it for y-axis
        if columns_found:
            y_col = columns_found[0]
            # Try to find a second column for x-axis
            x_col_candidates = [col for col in df.columns if col != y_col and df[col].nunique() < 50]
            x_col = x_col_candidates[0] if x_col_candidates else df.index.name or 'index'
            return x_col, y_col
        
        # Default to first two columns if no matches found
        if len(df.columns) >= 2:
            return df.columns[0], df.columns[1]
        elif len(df.columns) == 1:
            return df.index.name or 'index', df.columns[0]
        else:
            return 'index', 'value'
    
    def _create_bar_chart(self, df: pd.DataFrame, x_col: str, y_col: str, **kwargs) -> go.Figure:
        """Create a bar chart."""
        title = kwargs.get('title', f'{y_col} by {x_col}')
        fig = px.bar(df, x=x_col, y=y_col, title=title)
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
        return fig
    
    def _create_line_chart(self, df: pd.DataFrame, x_col: str, y_col: str, **kwargs) -> go.Figure:
        """Create a line chart."""
        title = kwargs.get('title', f'{y_col} over {x_col}')
        fig = px.line(df, x=x_col, y=y_col, title=title)
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
        return fig
    
    def _create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, **kwargs) -> go.Figure:
        """Create a scatter plot."""
        title = kwargs.get('title', f'Relationship between {x_col} and {y_col}')
        fig = px.scatter(df, x=x_col, y=y_col, title=title)
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
        return fig
    
    def _create_pie_chart(self, df: pd.DataFrame, x_col: str, y_col: str, **kwargs) -> go.Figure:
        """Create a pie chart."""
        title = kwargs.get('title', f'Distribution of {y_col}')
        fig = px.pie(df, names=x_col, values=y_col, title=title)
        return fig
    
    def _create_histogram(self, df: pd.DataFrame, x_col: str, **kwargs) -> go.Figure:
        """Create a histogram."""
        title = kwargs.get('title', f'Distribution of {x_col}')
        fig = px.histogram(df, x=x_col, title=title)
        fig.update_layout(xaxis_title=x_col, yaxis_title='Count')
        return fig
    
    def _create_box_plot(self, df: pd.DataFrame, x_col: str, y_col: str, **kwargs) -> go.Figure:
        """Create a box plot."""
        title = kwargs.get('title', f'Distribution of {y_col} by {x_col}')
        fig = px.box(df, x=x_col, y=y_col, title=title)
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
        return fig
    
    def create_visualization(self, df: pd.DataFrame, user_input: str) -> Dict[str, Any]:
        """Create a visualization based on the data and user request.
        
        Args:
            df: The DataFrame containing the data
            user_input: The user's request for visualization
            
        Returns:
            Dict containing the visualization data
        """
        try:
            # Clean and prepare the data
            df = df.copy()
            
            # Reset index to include it as a column if needed
            if df.index.name is not None:
                df = df.reset_index()
            
            # Detect chart type from user input
            chart_type = self.detect_chart_type(user_input)
            
            # Get appropriate columns for the chart
            x_col, y_col = self._extract_columns(df, user_input)
            
            # Generate the appropriate chart
            if chart_type in self.supported_charts:
                if chart_type == 'histogram':
                    fig = self._create_histogram(df, x_col, title=user_input)
                else:
                    fig = self.supported_charts[chart_type](df, x_col, y_col, title=user_input)
                
                # Convert to dictionary for serialization
                return {
                    'type': 'plotly',
                    'data': fig.to_dict(),
                    'chart_type': chart_type,
                    'title': user_input[:100]  # Truncate title if too long
                }
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")
                
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            raise

# Global instance
viz_generator = VisualizationGenerator()
