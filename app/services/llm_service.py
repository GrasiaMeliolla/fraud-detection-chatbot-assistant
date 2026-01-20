"""
LLM service for OpenAI integration.
"""
from typing import List, Dict, Optional, AsyncGenerator
from openai import AsyncOpenAI
from app.config import settings
import logging
import json

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM operations using OpenAI."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text completion from prompt."""
        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            logger.info(f"LLM generated {len(content)} characters")
            return content

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise

    async def generate_with_context(
        self,
        user_message: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> str:
        """Generate response with additional context."""
        prompt = user_message

        if context:
            prompt = f"""Context:
{context}

User Question:
{user_message}

Please answer the question using the provided context."""

        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> AsyncGenerator[str, None]:
        """Stream text completion from prompt."""
        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error in streaming LLM response: {e}")
            raise

    async def extract_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ):
        """Generate and extract JSON (dict or list) from LLM response."""
        try:
            response = await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
            )

            # Try to parse JSON directly
            try:
                return json.loads(response.strip())
            except json.JSONDecodeError:
                # Try to extract JSON object or array from response
                import re
                # Try array first
                array_match = re.search(r'\[.*\]', response, re.DOTALL)
                if array_match:
                    try:
                        return json.loads(array_match.group())
                    except:
                        pass
                # Try object
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except:
                        pass
                logger.error(f"Could not extract JSON from response: {response}")
                return {}

        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
            return {}


# Singleton instance
llm_service = LLMService()
