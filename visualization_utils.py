import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Union
import re
import logging

logger = logging.getLogger(_name_)

class VisualizationGenerator:
    """
    A class to generate various types of visualizations from pandas DataFrames.
    """
    
    @staticmethod
    def detect_visualization_type(query: str) -> Optional[str]:
        """
        Detect the type of visualization requested in the user's query.
        
        Args:
            query: User's query string
            
        Returns:
            str: Type of visualization (bar, line, scatter, etc.) or None if not detected
        """
        query = query.lower()
        
        visualization_map = {
            'bar': ['bar', 'column', 'barchart', 'bar chart', 'bar graph'],
            'line': ['line', 'trend', 'time series', 'line chart'],
            'scatter': ['scatter', 'scatterplot', 'scatter plot', 'correlation'],
            'pie': ['pie', 'donut', 'pie chart', 'donut chart'],
            'histogram': ['histogram', 'distribution', 'frequency'],
            'box': ['box', 'boxplot', 'box plot', 'whisker'],
            'heatmap': ['heatmap', 'heat map', 'correlation matrix', 'correlation heatmap']
        }
        
        for viz_type, keywords in visualization_map.items():
            if any(keyword in query for keyword in keywords):
                return viz_type
                
        return None
    
    @staticmethod
    def extract_columns(query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract column names from the query that match the DataFrame columns.
        
        Args:
            query: User's query string
            df: The DataFrame
        Returns:
            Dict containing x, y, color, and other relevant parameters
        """
        result = {'x': None, 'y': None, 'color': None, 'agg': None}
        query_lower = query.lower()
        df_columns = df.columns.tolist()

        # Check for aggregation functions
        agg_functions = ['sum', 'average', 'mean', 'count', 'max', 'min', 'median']
        for func in agg_functions:
            if func in query_lower:
                result['agg'] = func
                break

        # Match column names from the query
        for col in df_columns:
            col_lower = col.lower()
            # Check for x-axis
            if col_lower in query_lower and not result.get('x'):
                result['x'] = col
            # Check for y-axis (if mentioned)
            elif col_lower in query_lower and not result.get('y'):
                result['y'] = col
            # Check for color/hue
            if f"color by {col_lower}" in query_lower or f"group by {col_lower}" in query_lower:
                result['color'] = col

        # Enhanced fallback: infer numeric columns by checking actual values, not just dtype
        import pandas as pd
        inferred_numeric_cols = []
        for col in df.columns:
            # If already numeric, accept
            if pd.api.types.is_numeric_dtype(df[col]):
                inferred_numeric_cols.append(col)
            # If object, try to convert
            elif df[col].dtype == 'object':
                converted = pd.to_numeric(df[col], errors='coerce')
                # If more than 80% of non-null values convert, treat as numeric
                non_null = df[col].notnull().sum()
                numeric_count = converted.notnull().sum()
                if non_null > 0 and numeric_count / non_null > 0.8:
                    inferred_numeric_cols.append(col)
        # Use inferred numeric columns for fallback
        if (not result.get('x') or not result.get('y')) and inferred_numeric_cols:
            if not result.get('x') and len(inferred_numeric_cols) > 0:
                result['x'] = inferred_numeric_cols[0]
            if not result.get('y') and len(inferred_numeric_cols) > 1:
                result['y'] = inferred_numeric_cols[1]
            elif not result.get('y') and len(inferred_numeric_cols) == 1:
                result['y'] = inferred_numeric_cols[0]
        # If still missing, fallback to first two columns
        if (not result.get('x') or not result.get('y')) and len(df_columns) >= 2:
            if not result.get('x'):
                result['x'] = df_columns[0]
            if not result.get('y'):
                result['y'] = df_columns[1]
        elif (not result.get('x') or not result.get('y')) and len(df_columns) == 1:
            if not result.get('x'):
                result['x'] = df_columns[0]
            if not result.get('y'):
                result['y'] = df_columns[0]
        return result
    
    def create_visualization(self, df: pd.DataFrame, query: str) -> go.Figure:
        """
        Create a visualization based on the user's query.
        
        Args:
            df: Input DataFrame
            query: User's query string
            
        Returns:
            plotly.graph_objects.Figure: The generated figure
        """
        viz_type = self.detect_visualization_type(query)
        if not viz_type:
            viz_type = 'bar'  # Default to bar chart

        params = self.extract_columns(query, df)

        try:
            if viz_type == 'bar':
                return self._create_bar_chart(df, **params)
            elif viz_type == 'line':
                return self._create_line_chart(df, **params)
            elif viz_type == 'scatter':
                return self._create_scatter_plot(df, **params)
            elif viz_type == 'pie':
                return self._create_pie_chart(df, **params)
            elif viz_type == 'histogram':
                return self._create_histogram(df, **params)
            elif viz_type == 'box':
                return self._create_box_plot(df, **params)
            elif viz_type == 'heatmap':
                return self._create_heatmap(df, **params)
            else:
                return self._create_bar_chart(df, **params)  # Default to bar chart
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            raise ValueError(f"I couldn't create the visualization: {str(e)}")
    
    def _create_bar_chart(self, df: pd.DataFrame, x: str, y: str = None, color: str = None, **kwargs) -> go.Figure:
        """Create a bar chart."""
        if not x and not y:
            raise ValueError("Please specify at least one column for the x or y axis.")
            
        if not y:  # If only x is provided, count the values
            fig = px.bar(df[x].value_counts().reset_index(), 
                        x='index', y=x, 
                        title=f"Count of {x}")
        else:
            fig = px.bar(df, x=x, y=y, color=color,
                        title=f"{y} by {x}" + (f" (grouped by {color})" if color else ""))
        
        fig.update_layout(
            xaxis_title=x if x else "",
            yaxis_title=y if y else "Count",
            showlegend=bool(color)
        )
        return fig
    
    def _create_line_chart(self, df: pd.DataFrame, x: str, y: str, color: str = None, **kwargs) -> go.Figure:
        """Create a line chart."""
        if not x or not y:
            raise ValueError("Please specify both x and y columns for a line chart.")
            
        fig = px.line(df, x=x, y=y, color=color,
                     title=f"{y} over {x}" + (f" (by {color})" if color else ""))
        
        fig.update_layout(
            xaxis_title=x,
            yaxis_title=y,
            showlegend=bool(color)
        )
        return fig
    
    def _create_scatter_plot(self, df: pd.DataFrame, x: str, y: str, color: str = None, **kwargs) -> go.Figure:
        """Create a scatter plot."""
        if not x or not y:
            raise ValueError("Please specify both x and y columns for a scatter plot.")
            
        fig = px.scatter(df, x=x, y=y, color=color,
                        title=f"{y} vs {x}" + (f" (by {color})" if color else ""),
                        trendline="ols" if len(df) > 10 else None)
        
        fig.update_layout(
            xaxis_title=x,
            yaxis_title=y,
            showlegend=bool(color)
        )
        return fig
    
    def _create_pie_chart(self, df: pd.DataFrame, x: str, y: str = None, **kwargs) -> go.Figure:
        """Create a pie or donut chart."""
        if not x:
            raise ValueError("Please specify a column for the pie chart.")
            
        if not y:  # If only x is provided, count the values
            value_counts = df[x].value_counts().reset_index()
            fig = px.pie(value_counts, values=x, names='index', 
                         title=f"Distribution of {x}")
        else:
            fig = px.pie(df, values=y, names=x, 
                         title=f"{y} by {x}")
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    
    def _create_histogram(self, df: pd.DataFrame, x: str, color: str = None, **kwargs) -> go.Figure:
        """Create a histogram."""
        if not x:
            raise ValueError("Please specify a column for the histogram.")
            
        fig = px.histogram(df, x=x, color=color,
                          title=f"Distribution of {x}" + (f" (by {color})" if color else ""),
                          marginal="box")
        
        fig.update_layout(
            xaxis_title=x,
            yaxis_title="Count",
            showlegend=bool(color)
        )
        return fig
    
    def _create_box_plot(self, df: pd.DataFrame, x: str, y: str = None, color: str = None, **kwargs) -> go.Figure:
        """Create a box plot."""
        if not x and not y:
            raise ValueError("Please specify at least one column for the box plot.")
            
        if not y:  # If only x is provided, create a single box plot
            fig = px.box(df, y=x, title=f"Box plot of {x}")
            x_title = ""
            y_title = x
        else:
            fig = px.box(df, x=x, y=y, color=color,
                        title=f"Box plot of {y} by {x}" + (f" (grouped by {color})" if color else ""))
            x_title = x
            y_title = y
        
        fig.update_layout(
            xaxis_title=x_title,
            yaxis_title=y_title,
            showlegend=bool(color)
        )
        return fig
    
    def _create_heatmap(self, df: pd.DataFrame, **kwargs) -> go.Figure:
        """Create a correlation heatmap."""
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        if len(numeric_df.columns) < 2:
            raise ValueError("Need at least two numeric columns for a heatmap.")
            
        corr = numeric_df.corr()
        
        fig = px.imshow(corr,
                       labels=dict(x="", y="", color="Correlation"),
                       x=corr.columns,
                       y=corr.columns,
                       title="Correlation Heatmap")
        
        # Add correlation values to the heatmap
        fig.update_xaxes(side="bottom")
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Correlation",
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
                yanchor="top", y=1,
                xanchor="right", x=1.1
            )
        )
        
        return fig

# Global instance
viz_generator = VisualizationGenerator()