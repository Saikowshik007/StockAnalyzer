import re
from telegram._bot import Bot
from telegram.constants import ParseMode

class Notifier:
    def __init__(self):
        self.token = '7848324010:AAFpiHM5wVKr4JZfSyDMVadz1Cr5q-qJoA0'

    async def send_telegram_message(self, news):
        bot = Bot(token=self.token)

        formatted_message = self.format_summary_for_telegram(news)

        await bot.send_message(
            chat_id='-4606618937',
            text=formatted_message,
            parse_mode=ParseMode.MARKDOWN_V2
        )

    def format_summary_for_telegram(self, news):
        """Format summary into a visually appealing Telegram message with proper Markdown formatting"""
        # Extract data
        if hasattr(news, 'article'):
            # Object is a News instance
            title = news.article.title
            summary = news.summary
            url = news.article.url
        else:
            # Object might be a summary string + URL
            summary = news
            url_match = re.search(r'Reference link: (https?://\S+)', summary)
            if url_match:
                url = url_match.group(1)
                summary = summary.replace(f"Reference link: {url}", "")
            else:
                url = ""

            # Try to extract title from summary or use default
            title = "Financial News Update"

        # Clean up the summary by removing markdown-like elements from the original
        summary = self._clean_gpt_formatting(summary)

        # Escape special characters for Markdown V2
        title_escaped = self._escape_markdown(title)

        # Extract sentiment data
        sentiment = self._extract_sentiment(summary)
        rating = self._extract_rating(summary)
        reasoning = self._extract_reasoning(summary)

        # Extract implications and insights
        implications = self._extract_market_implications(summary)
        insights = self._extract_actionable_insights(summary)

        # Build the formatted message
        formatted_message = (
            f"📊 *FINANCIAL NEWS ALERT* 📊\n\n"
            f"📰 *{title_escaped}* 📰\n\n"
            f"*SENTIMENT:* {self._escape_markdown(sentiment)}\n"
            f"*Rating:* {self._escape_markdown(rating)}/10\n"
            f"*Reasoning:* {self._escape_markdown(reasoning)}\n\n"
            f"*KEY MARKET IMPLICATIONS:*\n{self._format_bullet_list(implications)}\n\n"
            f"*ACTIONABLE INSIGHTS:*\n{self._format_insights_list(insights)}\n\n"
        )

        # Add URL if available
        if url:
            escaped_url = self._escape_markdown(url)
            formatted_message += f"🔗 [Read Full Article]({escaped_url})"

        return formatted_message

    def _clean_gpt_formatting(self, text):
        """Remove GPT's markdown-style formatting from the text"""
        # Remove double asterisks that GPT uses for bold
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        return text

    def _escape_markdown(self, text):
        """Escape special characters for Markdown V2"""
        if not text:
            return ""

        # These characters need to be escaped in Markdown V2
        escape_chars = r'_*[]()~`>#+-=|{}.!'
        for char in escape_chars:
            text = text.replace(char, f"\\{char}")
        return text

    def _extract_sentiment(self, summary):
        """Extract sentiment category"""
        if "Sentiment Category:" in summary:
            match = re.search(r'Sentiment Category:(.+?)(?:\n|$)', summary)
            if match:
                return match.group(1).strip()

        # If not found with the specific label, try to find any sentiment mention
        match = re.search(r'Sentiment:(.+?)(?:\n|$)', summary)
        if match:
            return match.group(1).strip()

        return "Neutral"

    def _extract_rating(self, summary):
        """Extract numerical rating"""
        if "Numerical Rating:" in summary:
            match = re.search(r'Numerical Rating:(.+?)(?:\n|$)', summary)
            if match:
                # Extract just the number
                rating_text = match.group(1).strip()
                # Try to get just the number
                number_match = re.search(r'(\d+)', rating_text)
                if number_match:
                    return number_match.group(1)
                return rating_text

        # Alternative pattern
        match = re.search(r'Rating:(.+?)(?:\n|$)', summary)
        if match:
            rating_text = match.group(1).strip()
            number_match = re.search(r'(\d+)', rating_text)
            if number_match:
                return number_match.group(1)
            return rating_text

        return "N/A"

    def _extract_reasoning(self, summary):
        """Extract reasoning behind sentiment"""
        if "Reasoning:" in summary:
            match = re.search(r'Reasoning:(.+?)(?:\n|$|\.\s+\S)', summary)
            if match:
                reasoning = match.group(1).strip()
                # Truncate if too long
                if len(reasoning) > 180:
                    reasoning = reasoning[:177] + "..."
                return reasoning
        return "See full analysis for details"

    def _extract_market_implications(self, summary):
        """Extract market implications as a list"""
        implications = []

        # Try different patterns to extract implications
        if "KEY MARKET IMPLICATIONS:" in summary:
            section = summary.split("KEY MARKET IMPLICATIONS:")[1]
            if "ACTIONABLE INSIGHTS:" in section:
                section = section.split("ACTIONABLE INSIGHTS:")[0]

            # Look for numbered points with various formats
            patterns = [
                r'\d+\.\s+(?:\*\*)?([^:*]+)(?::\*\*|\*\*|:)',  # Numbered points with asterisks
                r'•\s+(?:\*\*)?([^:*]+)(?::\*\*|\*\*|:)',      # Bullet points with asterisks
                r'\n\s*([A-Z][^:]+):'                          # Capital letter starts, colon ends
            ]

            for pattern in patterns:
                matches = re.findall(pattern, section)
                if matches:
                    implications.extend([m.strip() for m in matches])
                    break

        # If no implications found with patterns, try to extract any sentence-like content
        if not implications and "KEY MARKET IMPLICATIONS:" in summary:
            section = summary.split("KEY MARKET IMPLICATIONS:")[1]
            if "ACTIONABLE INSIGHTS:" in section:
                section = section.split("ACTIONABLE INSIGHTS:")[0]

            # Extract sentences that might be implications
            sentences = re.findall(r'([^.\n]+\.[^.\n]+\.)', section)
            implications = [s.strip() for s in sentences[:3]]  # Take up to 3 sentences

        # Default if nothing found
        if not implications:
            implications = ["See full analysis for details"]

        return implications

    def _extract_actionable_insights(self, summary):
        """Extract actionable insights as a list"""
        insights = []

        if "ACTIONABLE INSIGHTS:" in summary:
            section = summary.split("ACTIONABLE INSIGHTS:")[1]
            if "Reference link:" in section:
                section = section.split("Reference link:")[0]

            # Similar patterns as implications
            patterns = [
                r'\d+\.\s+(?:\*\*)?([^:*]+)(?::\*\*|\*\*|:)',
                r'•\s+(?:\*\*)?([^:*]+)(?::\*\*|\*\*|:)',
                r'\n\s*([A-Z][^:]+):'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, section)
                if matches:
                    insights.extend([m.strip() for m in matches])
                    break

        if not insights:
            insights = ["See full analysis for details"]

        return insights

    def _format_bullet_list(self, items):
        """Format a list of items as bullet points with proper Markdown escaping"""
        if not items:
            return "• See full analysis for details"

        result = ""
        for item in items:
            escaped_item = self._escape_markdown(item)
            result += f"• *{escaped_item}*\n"
        return result

    def _format_insights_list(self, items):
        """Format a list of actionable insights with emoji and proper Markdown escaping"""
        if not items:
            return "💡 See full analysis for details"

        result = ""
        for item in items:
            escaped_item = self._escape_markdown(item)
            result += f"💡 *{escaped_item}*\n"
        return result