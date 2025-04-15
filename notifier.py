import os
import re
from telegram._bot import Bot
from telegram.constants import ParseMode


class Notifier:
    def __init__(self):
        self.token = os.environ.get("TELEGRAM_API_KEY")


    async def send_telegram_message(self, news):
        bot = Bot(token=self.token)
        formatted_message = self.format_summary_for_telegram(news)
        await bot.send_message(
            chat_id='-4606618937',
            text=formatted_message,
            parse_mode=ParseMode.MARKDOWN_V2
        )

    def _extract_sentiment_category(self, summary):
        """Extract sentiment category with improved pattern matching"""
        # Match both with and without asterisks
        match = re.search(r'\*\*Sentiment Category:\*\*\s*([^()\n]+)(?:\([\d-]+/10\))?', summary)
        if not match:
            match = re.search(r'Sentiment Category:\s*([^()\n]+)(?:\([\d-]+/10\))?', summary, re.IGNORECASE)

        if match:
            return match.group(1).strip()

        return "Neutral"

    def _extract_rating(self, summary):
        """Extract numerical rating with improved pattern matching"""
        # Try to match with asterisks
        match = re.search(r'\*\*Numerical Rating:\*\*\s*(\d+)', summary)
        if not match:
            match = re.search(r'Numerical Rating:\s*(\d+)', summary, re.IGNORECASE)

        # Try to extract from sentiment category with rating
        if not match:
            match = re.search(r'Sentiment Category:.*?\((\d+)(?:-\d+)?/10\)', summary, re.IGNORECASE)

        if match:
            return match.group(1).strip()

        return "5"  # Default neutral rating

    def _extract_reasoning(self, summary):
        """Extract reasoning behind sentiment with improved pattern matching"""
        # Try with asterisks
        match = re.search(r'\*\*Reasoning:\*\*\s*(.*?)(?=\d+\.\s+\*\*KEY|\*\*KEY|KEY)', summary, re.DOTALL)
        if not match:
            match = re.search(r'Reasoning:\s*(.*?)(?=\d+\.\s+\*\*KEY|\*\*KEY|KEY)', summary, re.DOTALL | re.IGNORECASE)

        if match:
            reasoning = match.group(1).strip()
            # Truncate if too long but at a sensible point (end of sentence)
            if len(reasoning) > 200:
                last_period = reasoning[:197].rfind('.')
                if last_period > 150:  # Make sure we have a decent length
                    reasoning = reasoning[:last_period+1]
                else:
                    reasoning = reasoning[:197] + "..."
            return reasoning

        return "See full analysis for details"

    def _extract_market_implications(self, summary):
        """Extract market implications as a list with improved pattern matching"""
        implications = []

        # Try to match the nested structure by first getting the whole KEY MARKET IMPLICATIONS section
        section_match = re.search(r'(?:\d+\.\s+)?\*\*KEY MARKET IMPLICATIONS:\*\*(.*?)(?=\d+\.\s+\*\*ACTIONABLE|\*\*ACTIONABLE|ACTIONABLE)',
                                  summary, re.DOTALL)

        if not section_match:
            section_match = re.search(r'(?:\d+\.\s+)?KEY MARKET IMPLICATIONS:(.*?)(?=\d+\.\s+\*\*ACTIONABLE|\*\*ACTIONABLE|ACTIONABLE)',
                                      summary, re.DOTALL | re.IGNORECASE)

        if section_match:
            section = section_match.group(1)

            # Extract the main points - look for Point patterns, Impact patterns, etc.
            point_matches = re.findall(r'\*\*Point\s+\d+:\*\*\s*(.*?)(?=\s+[-•]\s+\*\*Impact|\s+\*\*Impact|$)',
                                       section, re.DOTALL | re.IGNORECASE)

            if not point_matches:
                point_matches = re.findall(r'Point\s+\d+:\s*(.*?)(?=\s+[-•]\s+Impact|\s+Impact|$)',
                                           section, re.DOTALL | re.IGNORECASE)

            # If still no matches, look for bullet points
            if not point_matches:
                bullet_matches = re.findall(r'[-•]\s+\*\*([^:*]+)(?:\*\*)?:(.*?)(?=[-•]|$)', section, re.DOTALL)
                point_matches = [f"{header}: {content.strip()}" for header, content in bullet_matches]

            # If still nothing, just grab any bullet points
            if not point_matches:
                point_matches = re.findall(r'[-•]\s+(.*?)(?=[-•]|$)', section, re.DOTALL)

            implications = [p.strip() for p in point_matches if p.strip()]

        # If no implications found, use default
        if not implications:
            implications = ["Market impact analysis available in full report"]

        return implications[:5]  # Limit to 5 points

    def _extract_actionable_insights(self, summary):
        """Extract actionable insights with improved pattern matching"""
        insights = []

        # Try to match the ACTIONABLE INSIGHTS section with asterisks
        section_match = re.search(r'(?:\d+\.\s+)?\*\*ACTIONABLE INSIGHTS:\*\*(.*?)(?=Reference link:|$)',
                                  summary, re.DOTALL)

        if not section_match:
            section_match = re.search(r'(?:\d+\.\s+)?ACTIONABLE INSIGHTS:(.*?)(?=Reference link:|$)',
                                      summary, re.DOTALL | re.IGNORECASE)

        if section_match:
            section = section_match.group(1)

            # Try to extract investment strategies with asterisks
            strategy_matches = re.findall(r'\*\*Investment Strategy\s+\d+:\*\*\s*(.*?)(?=\*\*Investment Strategy|\*\*Confidence|$)',
                                          section, re.DOTALL)

            if not strategy_matches:
                strategy_matches = re.findall(r'Investment Strategy\s+\d+:\s*(.*?)(?=Investment Strategy|Confidence|$)',
                                              section, re.DOTALL | re.IGNORECASE)

            # If no strategies found, try to extract bullet points
            if not strategy_matches:
                strategy_matches = re.findall(r'[-•]\s+(.*?)(?=[-•]|$)', section, re.DOTALL)

            insights = [s.strip() for s in strategy_matches if s.strip()]

        # If no insights found, use default
        if not insights:
            insights = ["Actionable insights available in full report"]

        return insights[:3]  # Limit to 3 insights

    def _clean_gpt_formatting(self, text):
        """Remove GPT's markdown-style formatting from the text"""
        # Remove double asterisks that are used for bold
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        # Remove numbering prefixes like "1. ", "2. ", etc.
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        # Remove bullet points
        text = re.sub(r'^\s*[-•]\s+', '', text, flags=re.MULTILINE)
        return text

    def format_summary_for_telegram(self, news):
        """Format summary into a visually appealing Telegram message with proper Markdown formatting"""
        # Extract data from the news object
        summary = news.summary
        title = news.article.title
        article_url = news.article.url  # Use a different variable name for the URL

        # Parse the summary text
        sentiment_category = self._extract_sentiment_category(summary)
        rating = self._extract_rating(summary)
        reasoning = self._extract_reasoning(summary)
        implications = self._extract_market_implications(summary)
        insights = self._extract_actionable_insights(summary)

        # Make sure to escape the title and all content properly
        title_escaped = self._escape_markdown(title)

        # Build message with proper newlines
        formatted_message = (
            f"📊 *FINANCIAL NEWS ALERT* 📊\n\n"
            f"📰 *{title_escaped}* 📰\n\n"
            f"*SENTIMENT:* {self._escape_markdown(sentiment_category)}\n"
            f"*Rating:* {self._escape_markdown(rating)}/10\n"
            f"*Reasoning:* {self._escape_markdown(reasoning)}\n\n"
            f"*KEY MARKET IMPLICATIONS:*\n{self._format_bullet_list(implications)}\n\n"
            f"*ACTIONABLE INSIGHTS:*\n{self._format_insights_list(insights)}\n\n"
        )

        # Add URL if available
        if article_url:  # Use a different variable name
            escaped_url = self._escape_markdown(article_url)
            formatted_message += f"🔗 [Read Full Article]({escaped_url})"

        return formatted_message

    def _escape_markdown(self, text):
        """Escape special characters for Telegram's Markdown V2 format"""
        if not text:
            return ""

        # These characters need to be escaped in Telegram's Markdown V2
        escape_chars = '_*[]()~`>#+-=|{}.!'

        # Create a new string with escaped characters
        escaped_text = ""
        for char in text:
            if char in escape_chars:
                escaped_text += f"\\{char}"
            else:
                escaped_text += char

        return escaped_text

    def _format_bullet_list(self, items):
        """Format a list of items as bullet points with proper Markdown escaping"""
        if not items or (len(items) == 1 and not items[0].strip()):
            return "• See full analysis for details"

        result = ""
        for item in items:
            escaped_item = self._escape_markdown(item)
            result += f"• *{escaped_item}*\n"
        return result

    def _format_insights_list(self, items):
        """Format a list of actionable insights with emoji and proper Markdown escaping"""
        if not items or (len(items) == 1 and not items[0].strip()):
            return "💡 See full analysis for details"

        result = ""
        for item in items:
            escaped_item = self._escape_markdown(item)
            result += f"💡 *{escaped_item}*\n"
        return result