import openai
from typing import List, Dict, Any
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)

class QuestionRewriter:
    """
    A class for rewriting questions to improve context retrieval.
    
    Attributes:
        model (str): OpenAI model used for question rewriting
    """
    
    def __init__(self):
        """
        Initialize the QuestionRewriter with OpenAI API configuration.
        """
        openai.api_key = settings.OPENAI_API_KEY
        self.model = "gpt-3.5-turbo"

    def rewrite_question(self, question: str) -> List[str]:
        """
        Generate multiple paraphrased versions of a question to improve context retrieval.
        
        Args:
            question (str): Original question to be rewritten
            
        Returns:
            List[str]: List of paraphrased questions
        """
        try:
            prompt = f"""Given the following question, rewrite it in different ways to help find relevant context:
            Original question: {question}
            
            Please provide 3-4 different versions of this question that maintain the same meaning but are worded differently.
            Format the response as a JSON array of strings."""

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that rewrites questions to help find relevant context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            rewritten_questions = response.choices[0].message.content
            return self._parse_rewritten_questions(rewritten_questions)
        except Exception as e:
            logger.error(f"Error rewriting question: {e}")
            return [question]

    def rewrite_for_context(self, question: str, context: str) -> List[str]:
        """
        Generate question variations that better match the provided context.
        
        Args:
            question (str): Original question
            context (str): Relevant context to consider
            
        Returns:
            List[str]: List of context-aware question variations
        """
        try:
            prompt = f"""Given the following question and context, rewrite the question to better match the context:
            Question: {question}
            Context: {context}
            
            Please provide 2-3 versions of the question that are more likely to find relevant information in the context.
            Format the response as a JSON array of strings."""

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that rewrites questions to better match context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )

            rewritten_questions = response.choices[0].message.content
            return self._parse_rewritten_questions(rewritten_questions)
        except Exception as e:
            logger.error(f"Error rewriting for context: {e}")
            return [question]

    def _parse_rewritten_questions(self, response: str) -> List[str]:
        """
        Parse the JSON response containing rewritten questions.
        
        Args:
            response (str): JSON string containing rewritten questions
            
        Returns:
            List[str]: List of rewritten questions
        """
        try:
            import json
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error("Invalid JSON response, returning original response")
            return [response]
        except Exception as e:
            logger.error(f"Error parsing rewritten questions: {e}")
            return []
