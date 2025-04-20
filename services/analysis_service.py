import re
from typing import Dict, List, Optional

class AnalysisService:
    """Parse and process summaries for database storage."""

    def __init__(self):
        pass

    def extract_components(self, summary: str) -> Dict:
        """Extract structured components from summary text."""
        components = {
            'sentiment_category': self._extract_sentiment_category(summary),
            'sentiment_rating': float(self._extract_rating(summary)),
            'reasoning': self._extract_reasoning(summary),
            'market_implications': self._extract_market_implications(summary),
            'actionable_insights': self._extract_actionable_insights(summary)
        }
        return components

    def _extract_sentiment_category(self, summary):
        """Extract sentiment category with improved pattern matching"""
        match = re.search(r'\*\*Sentiment Category:\*\*\s*([^()\n]+)(?:\([\d-]+/10\))?', summary)
        if not match:
            match = re.search(r'Sentiment Category:\s*([^()\n]+)(?:\([\d-]+/10\))?', summary, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "Neutral"

    def _extract_rating(self, summary):
        """Extract numerical rating with improved pattern matching"""
        match = re.search(r'\*\*Numerical Rating:\*\*\s*(\d+)', summary)
        if not match:
            match = re.search(r'Numerical Rating:\s*(\d+)', summary, re.IGNORECASE)
        if not match:
            match = re.search(r'Sentiment Category:.*?\((\d+)(?:-\d+)?/10\)', summary, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "5"

    def _extract_reasoning(self, summary):
        """Extract reasoning behind sentiment."""
        match = re.search(r'\*\*Reasoning:\*\*\s*(.*?)(?=\d+\.\s+\*\*KEY|\*\*KEY|KEY)', summary, re.DOTALL)
        if not match:
            match = re.search(r'Reasoning:\s*(.*?)(?=\d+\.\s+\*\*KEY|\*\*KEY|KEY)', summary, re.DOTALL | re.IGNORECASE)
        if match:
            reasoning = match.group(1).strip()
            if len(reasoning) > 200:
                last_period = reasoning[:197].rfind('.')
                if last_period > 150:
                    reasoning = reasoning[:last_period+1]
                else:
                    reasoning = reasoning[:197] + "..."
            return reasoning
        return "See full analysis for details"

    def _extract_market_implications(self, summary):
        """Extract market implications from summary."""
        implications = []
        section_match = re.search(r'(?:\d+\.\s+)?\*\*KEY MARKET IMPLICATIONS:\*\*(.*?)(?=\d+\.\s+\*\*ACTIONABLE|\*\*ACTIONABLE|ACTIONABLE)',
                                  summary, re.DOTALL)
        if not section_match:
            section_match = re.search(r'(?:\d+\.\s+)?KEY MARKET IMPLICATIONS:(.*?)(?=\d+\.\s+\*\*ACTIONABLE|\*\*ACTIONABLE|ACTIONABLE)',
                                      summary, re.DOTALL | re.IGNORECASE)
        if section_match:
            section = section_match.group(1)
            point_matches = re.findall(r'\*\*Point\s+\d+:\*\*\s*(.*?)(?=\s+[-•]\s+\*\*Impact|\s+\*\*Impact|$)',
                                       section, re.DOTALL | re.IGNORECASE)
            if not point_matches:
                point_matches = re.findall(r'Point\s+\d+:\s*(.*?)(?=\s+[-•]\s+Impact|\s+Impact|$)',
                                           section, re.DOTALL | re.IGNORECASE)
            if not point_matches:
                bullet_matches = re.findall(r'[-•]\s+\*\*([^:*]+)(?:\*\*)?:(.*?)(?=[-•]|$)', section, re.DOTALL)
                point_matches = [f"{header}: {content.strip()}" for header, content in bullet_matches]
            if not point_matches:
                point_matches = re.findall(r'[-•]\s+(.*?)(?=[-•]|$)', section, re.DOTALL)
            implications = [p.strip() for p in point_matches if p.strip()]

        if not implications:
            implications = ["Market impact analysis available in full report"]

        return implications[:5]

    def _extract_actionable_insights(self, summary):
        """Extract actionable insights from summary."""
        insights = []
        section_match = re.search(r'(?:\d+\.\s+)?\*\*ACTIONABLE INSIGHTS:\*\*(.*?)(?=Reference link:|$)',
                                  summary, re.DOTALL)
        if not section_match:
            section_match = re.search(r'(?:\d+\.\s+)?ACTIONABLE INSIGHTS:(.*?)(?=Reference link:|$)',
                                      summary, re.DOTALL | re.IGNORECASE)
        if section_match:
            section = section_match.group(1)
            strategy_matches = re.findall(r'\*\*Investment Strategy\s+\d+:\*\*\s*(.*?)(?=\*\*Investment Strategy|\*\*Confidence|$)',
                                          section, re.DOTALL)
            if not strategy_matches:
                strategy_matches = re.findall(r'Investment Strategy\s+\d+:\s*(.*?)(?=Investment Strategy|Confidence|$)',
                                              section, re.DOTALL | re.IGNORECASE)
            if not strategy_matches:
                strategy_matches = re.findall(r'[-•]\s+(.*?)(?=[-•]|$)', section, re.DOTALL)
            insights = [s.strip() for s in strategy_matches if s.strip()]

        if not insights:
            insights = ["Actionable insights available in full report"]

        return insights[:3]