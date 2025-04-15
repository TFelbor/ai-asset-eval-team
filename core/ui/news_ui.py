"""
News UI component for the AI Finance Dashboard.
This module provides a Streamlit UI for displaying financial news.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from core.api.news_client import NewsClient

def render_news_ui():
    """Render the news UI component."""
    st.markdown('<h2 class="subheader">📰 Financial News</h2>', unsafe_allow_html=True)

    # Add a button to return to the main dashboard
    if st.button("Return to Home", key="return_home_news"):
        # Reset all view flags to return to the landing page
        st.session_state.show_comparison = False
        st.session_state.show_docs = False
        st.session_state.show_backtesting = False
        st.session_state.show_news = False
        st.session_state.last_analysis = None
        st.rerun()

    # Create tabs for different news categories
    tabs = st.tabs(["Market News", "Asset-Specific News", "Trending Topics"])

    with tabs[0]:  # Market News tab
        render_market_news()

    with tabs[1]:  # Asset-Specific News tab
        render_asset_news()

    with tabs[2]:  # Trending Topics tab
        render_trending_topics()

def render_market_news():
    """Render the market news section."""
    st.markdown("### Latest Market News")

    # Add filters
    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox(
            "Category",
            ["All", "Business", "Economy", "Markets", "Technology", "Politics"],
            key="market_news_category"
        )

    with col2:
        time_range = st.selectbox(
            "Time Range",
            ["Today", "Last 3 Days", "Last Week", "Last Month"],
            key="market_news_time_range"
        )

    # Initialize news client
    news_client = NewsClient()

    # Get market news
    with st.spinner("Loading market news..."):
        try:
            # Convert time range to days
            days = 1  # Default to today
            if time_range == "Last 3 Days":
                days = 3
            elif time_range == "Last Week":
                days = 7
            elif time_range == "Last Month":
                days = 30

            # Get news articles
            articles = news_client.get_market_news(limit=10)

            # Filter by category if needed
            if category != "All":
                articles = [article for article in articles if category.lower() in article.get("category", "").lower()]

            # Display news articles
            display_news_articles(articles)

        except Exception as e:
            st.error(f"Error loading market news: {str(e)}")

def render_asset_news():
    """Render the asset-specific news section."""
    st.markdown("### Asset-Specific News")

    # Asset type and ticker input
    col1, col2 = st.columns(2)

    with col1:
        asset_type = st.selectbox(
            "Asset Type",
            ["Stock", "Cryptocurrency", "ETF", "REIT"],
            key="asset_news_type"
        )

    with col2:
        ticker = st.text_input("Ticker/Symbol", key="asset_news_ticker")

    # Search button
    search_button = st.button("Search News", key="asset_news_search")

    if search_button and ticker:
        # Initialize news client
        news_client = NewsClient()

        # Get asset-specific news
        with st.spinner(f"Loading news for {ticker}..."):
            try:
                # Get news articles based on asset type
                if asset_type.lower() == "stock":
                    articles = news_client.get_stock_news(ticker, limit=10)
                elif asset_type.lower() == "cryptocurrency":
                    articles = news_client.get_crypto_news(ticker, limit=10)
                elif asset_type.lower() == "etf":
                    articles = news_client.get_etf_news(ticker, limit=10)
                elif asset_type.lower() == "reit":
                    articles = news_client.get_reit_news(ticker, limit=10)
                else:
                    articles = []

                # Display news articles
                if articles:
                    display_news_articles(articles)
                else:
                    st.info(f"No news found for {ticker}.")

            except Exception as e:
                st.error(f"Error loading news for {ticker}: {str(e)}")
    else:
        st.info("Enter a ticker/symbol and click 'Search News' to find asset-specific news.")

def render_trending_topics():
    """Render the trending topics section."""
    st.markdown("### Trending Financial Topics")

    # Initialize news client
    news_client = NewsClient()

    # Get trending topics
    with st.spinner("Analyzing trending topics..."):
        try:
            # Get market news
            articles = news_client.get_market_news(limit=30)

            # Extract and count topics
            topics = extract_trending_topics(articles)

            # Display trending topics
            if topics:
                # Create a bar chart of trending topics
                fig = px.bar(
                    x=list(topics.keys()),
                    y=list(topics.values()),
                    labels={"x": "Topic", "y": "Mentions"},
                    title="Trending Financial Topics",
                    color=list(topics.values()),
                    color_continuous_scale="Viridis"
                )

                # Update layout
                fig.update_layout(
                    height=400,
                    template="plotly_dark",
                    xaxis_title="Topic",
                    yaxis_title="Mentions"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display articles for each topic
                st.markdown("### Articles by Topic")

                # Create tabs for top topics
                top_topics = list(topics.keys())[:5]  # Top 5 topics
                if top_topics:
                    topic_tabs = st.tabs(top_topics)

                    for i, topic in enumerate(top_topics):
                        with topic_tabs[i]:
                            # Filter articles by topic with proper None handling
                            topic_articles = []
                            for article in articles:
                                # Safely get text fields with fallbacks to empty string if None
                                title = article.get("title", "") or ""
                                description = article.get("description", "") or ""
                                content = article.get("content", "") or ""

                                # Convert to lowercase
                                title = title.lower()
                                description = description.lower()
                                content = content.lower()

                                # Check if topic is in any of the text fields
                                if (topic.lower() in title or
                                    topic.lower() in description or
                                    topic.lower() in content):
                                    topic_articles.append(article)

                            # Display articles
                            display_news_articles(topic_articles[:5])  # Show top 5 articles
            else:
                st.info("No trending topics found.")

        except Exception as e:
            st.error(f"Error analyzing trending topics: {str(e)}")

def display_news_articles(articles):
    """Display news articles in a nice format."""
    if not articles:
        st.info("No articles found.")
        return

    # Display each article
    for i, article in enumerate(articles):
        with st.container():
            # Create columns for image and content
            col1, col2 = st.columns([1, 3])

            with col1:
                # Display image if available
                image_url = article.get("urlToImage") or article.get("image_url")
                if image_url:
                    st.image(image_url, use_container_width=True)
                else:
                    # Display a placeholder
                    st.markdown("📰")

            with col2:
                # Display article title and source
                st.markdown(f"### {article.get('title', 'No Title')}")
                st.markdown(f"**Source:** {article.get('source', {}).get('name', article.get('source', 'Unknown'))} | **Published:** {format_date(article.get('publishedAt', article.get('published_at', '')))}")

                # Display description
                description = article.get("description") or article.get("summary", "No description available.")
                st.markdown(description)

                # Add a link to the full article
                url = article.get("url", "#")
                st.markdown(f"[Read full article]({url})")

            # Add a separator
            st.markdown("---")

def format_date(date_str):
    """Format a date string for display."""
    if not date_str:
        return "Unknown date"

    try:
        # Try to parse the date string
        if isinstance(date_str, str):
            if date_str.endswith('Z'):
                date_str = date_str[:-1]  # Remove 'Z' suffix if present

            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))

            # Format the date
            now = datetime.now()
            diff = now - date_obj

            if diff.days == 0:
                # Today
                hours = diff.seconds // 3600
                if hours == 0:
                    minutes = diff.seconds // 60
                    return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
                else:
                    return f"{hours} hour{'s' if hours != 1 else ''} ago"
            elif diff.days == 1:
                # Yesterday
                return "Yesterday"
            elif diff.days < 7:
                # This week
                return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
            else:
                # Older
                return date_obj.strftime("%B %d, %Y")
        else:
            return "Unknown date"
    except Exception:
        return date_str

def extract_trending_topics(articles):
    """Extract trending topics from news articles."""
    if not articles:
        return {}

    # Common financial topics to look for
    financial_topics = [
        "inflation", "recession", "interest rates", "fed", "federal reserve",
        "stock market", "bull market", "bear market", "crypto", "bitcoin",
        "ethereum", "blockchain", "nft", "ai", "artificial intelligence",
        "earnings", "ipo", "merger", "acquisition", "dividend",
        "tech stocks", "oil", "gold", "commodities", "treasury",
        "bonds", "yield", "gdp", "economy", "unemployment",
        "housing market", "real estate", "etf", "fund", "investment",
        "banking", "fintech", "regulation", "tax", "stimulus"
    ]

    # Count mentions of each topic
    topic_counts = {topic: 0 for topic in financial_topics}

    for article in articles:
        # Safely get text fields with fallbacks to empty string if None
        title = article.get("title", "") or ""
        description = article.get("description", "") or ""
        content = article.get("content", "") or ""

        # Convert to lowercase
        title = title.lower()
        description = description.lower()
        content = content.lower()

        # Combine all text
        all_text = f"{title} {description} {content}"

        # Count mentions
        for topic in financial_topics:
            if topic in all_text:
                topic_counts[topic] += 1

    # Remove topics with no mentions
    topic_counts = {topic: count for topic, count in topic_counts.items() if count > 0}

    # Sort by count (descending)
    topic_counts = dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True))

    return topic_counts
