class Prompts:

    def __init__(self):
        pass

    def get_summarization_prompt(self):
        prompt = f'''As a seasoned financial market analyst with expertise in sentiment detection of the financial market news, your task is to evaluate the following summarized financial news and provide a comprehensive analysis.

            Please analyze the text and deliver:
            
            1. OVERALL SENTIMENT ASSESSMENT:
               * Categorize the sentiment as one of the following:
                 - Very Negative (1-3/10)
                 - Somewhat Negative (4-5/10)
                 - Neutral (6/10)
                 - Somewhat Positive (7-8/10)
                 - Very Positive (9-10/10)
               * Provide a specific numerical rating (1-10)
               * Explain your reasoning for this rating in 2-3 sentences
            
            2. KEY MARKET IMPLICATIONS:
               * Identify 3-5 significant points from the text
               * For each point, explain:
                 - The potential impact on relevant market sectors
                 - Any specific companies or industries mentioned
                 - Timeframe of potential impact (immediate, short-term, long-term)
            
            3. ACTIONABLE INSIGHTS:
               * Based on this news, suggest 1-2 potential investment strategies or considerations
               * Indicate confidence level in your assessment (low, medium, high)
            
            Please maintain objectivity and focus on factual analysis rather than speculation.
            
            Text to analyze:
            '''
        return prompt