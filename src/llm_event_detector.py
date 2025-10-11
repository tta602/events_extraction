import os
from typing import List, Tuple, Optional
from openai import OpenAI
import json

class LLMEventDetector:
    """
    Zero-shot event type detection using LLM.
    Can detect both known event types and suggest new ones.
    """
    
    def __init__(self, api_key: Optional[str] = None, known_event_types: List[str] = None, model: str = "gpt-4o-mini"):
        """
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            known_event_types: List of known event types for reference
            model: OpenAI model to use (gpt-4o-mini is cheap and fast)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.known_event_types = known_event_types or []
        
        # Cache for sentences already processed
        self.cache = {}
    
    def detect_event_types(self, sentence: str, top_k: int = 3, confidence_threshold: float = 0.7) -> List[Tuple[str, float, bool]]:
        """
        Detect event types in a sentence using LLM.
        
        Args:
            sentence: Input sentence
            top_k: Number of event types to return
            confidence_threshold: Minimum confidence to return
            
        Returns:
            List of (event_type, confidence, is_new) tuples
            is_new=True means this is a newly suggested event type not in known_event_types
        """
        # Check cache first
        cache_key = f"{sentence}_{top_k}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Create prompt
        prompt = self._create_prompt(sentence, top_k)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert event extraction system. Analyze text and identify event types."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            # Process results
            detected_events = []
            for event in result.get("events", [])[:top_k]:
                event_type = event.get("type", "")
                confidence = event.get("confidence", 0.0)
                
                if confidence >= confidence_threshold:
                    # Check if it's a new event type
                    is_new = event_type not in self.known_event_types
                    detected_events.append((event_type, confidence, is_new))
            
            # Cache result
            self.cache[cache_key] = detected_events
            
            return detected_events
            
        except Exception as e:
            print(f"[LLM ERROR] Failed to detect events: {e}")
            return []
    
    def _create_prompt(self, sentence: str, top_k: int) -> str:
        """Create prompt for LLM"""
        
        known_types_str = "\n".join([f"  - {et}" for et in self.known_event_types[:20]])  # Show first 20 as examples
        
        prompt = f"""Analyze this sentence and identify the top {top_k} most relevant event types.

SENTENCE: "{sentence}"

KNOWN EVENT TYPES (for reference, but you can suggest new ones):
{known_types_str}
... and {len(self.known_event_types) - 20} more types

EVENT TYPE FORMAT: Follow the pattern "Category.Subcategory.Detail"
Examples: 
- Life.Die.Unspecified
- Conflict.Attack.DetonateExplode
- Transaction.Transaction.Unspecified

INSTRUCTIONS:
1. If the event matches a known type, use that exact name
2. If it's a new event type not in the list, create a new type name following the format
3. For each event, provide:
   - type: The event type name
   - confidence: A float between 0.0 and 1.0
   - reasoning: Brief explanation

Return your answer as JSON:
{{
  "events": [
    {{
      "type": "Category.Subcategory.Detail",
      "confidence": 0.95,
      "reasoning": "Brief explanation"
    }}
  ]
}}"""
        
        return prompt
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache = {}