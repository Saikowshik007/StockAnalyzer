import os
from datetime import time

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from openai import OpenAI
import logging
from typing import  Optional

class Summarizer:
    def __init__(self, api_key: Optional[str] = None, max_tokens: int = 1024, model_name: str = "ProsusAI/finbert"):
        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Initialize FinBERT
        try:
            # Load with optimized parameters for accuracy while managing memory
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.logger.info(f"Model loaded on {self.device}")
        except Exception as e:
            self.logger.error(f"Error loading FinBERT model: {e}")
            raise

        # Initialize OpenAI client
        try:
            self.client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY")
            )
        except Exception as e:
            self.logger.error(f"Error initializing OpenAI client: {e}")
            raise

        # Set token length for transformer - balance between performance and memory
        self.max_tokens = max_tokens

        # Ensure NLTK resources are available
        self._ensure_nltk_resources()

    def _ensure_nltk_resources(self):
        """Make sure necessary NLTK resources are downloaded."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer")
            nltk.download('punkt', quiet=True)

    def generate_summary(self, article, max_length: int = 15) -> str:
        """Generate a summary for the given article."""
        # Validate input
        if not hasattr(article, 'text_cleaned') or not article.text_cleaned:
            self.logger.warning("No cleaned text available in article")
            return "No article text available to summarize."

        text = article.text_cleaned

        # Generate summary with FinBERT
        try:
            summarized_text = self._summarize_with_finbert(text, max_length)
            if not summarized_text:
                self.logger.warning("FinBERT returned empty summary")
                return "Could not generate summary from article text."
        except Exception as e:
            self.logger.error(f"Error in FinBERT summarization: {e}")
            return f"Error generating summary: {e}"

        # Return summarized text or analyze with ChatGPT if needed
        try:
            return self._analyze_with_chatgpt(summarized_text)
            return summarized_text
        except Exception as e:
            self.logger.error(f"Error in ChatGPT analysis: {e}")
            return f"Generated summary but analysis failed: {e}"

    def _summarize_with_finbert(self, text: str, max_length: int = 15) -> str:
        """Summarize text using FinBERT for financial sentiment scoring with proper chunking."""
        # Tokenize the text into sentences
        sentences = sent_tokenize(text)
        if not sentences:
            self.logger.warning("No sentences found in text")
            return ""

        # Limit number of sentences for memory efficiency while maintaining accuracy
        max_sentences = 150  # Balance between completeness and memory usage
        if len(sentences) > max_sentences:
            self.logger.info(f"Limiting analysis to first {max_sentences} sentences")
            sentences = sentences[:max_sentences]

        # Process sentences in batches - smaller for memory constraints
        batch_size = 8  # Balance between speed and memory usage
        all_importance_scores = []

        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]

            # Tokenize sentences
            inputs = self.tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_tokens
            ).to(self.device)

            # Clear CUDA cache between batches if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Get model outputs without gradient calculation
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Calculate importance scores (variance of sentiment probabilities)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            batch_importance_scores = torch.var(scores, dim=1).cpu().numpy()
            all_importance_scores.extend(batch_importance_scores)

        # Convert to numpy array for processing
        all_importance_scores = np.array(all_importance_scores)

        # Get indices of top sentences by importance score
        top_length = min(max_length, len(sentences))
        top_idx = np.argsort(all_importance_scores)[-top_length:]

        # Sort indices to maintain original sentence order
        top_idx = sorted(top_idx)

        # Build summary from top sentences
        summary_sentences = [sentences[i] for i in top_idx]
        summary = " ".join(summary_sentences)

        return summary

    def _analyze_with_chatgpt(self, summary_text: str) -> str:
        """Use ChatGPT to analyze a financial text summary."""
        from Prompts import Prompts

        try:
            prompts = Prompts()

            # Create response with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",  # Use mini model for efficiency
                        messages=[
                            {"role": "system", "content": prompts.get_summarization_prompt()},
                            {"role": "user", "content": summary_text}
                        ],
                        temperature=0.1,  # Lower temperature for more deterministic results
                        max_tokens=1024   # Sufficient for analysis without excessive usage
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"OpenAI API attempt {attempt+1} failed: {e}, retrying...")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise

        except Exception as e:
            self.logger.error(f"Error using ChatGPT for analysis: {e}")
            return f"Analysis failed: {str(e)}"