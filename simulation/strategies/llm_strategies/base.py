import os
import json
import requests
import logging
from pathlib import Path
from dotenv import load_dotenv
from ..base import BaseStrategy
from ...cards import get_card_value
import weave

logger = logging.getLogger(__name__)

weave.init("dynastai")  # Set your project name here

class LLMStrategy(BaseStrategy):
    """
    Strategy that uses an LLM via OpenRouter API to make decisions.
    """
    def __init__(self, 
                 api_key=None, 
                 api_url='https://openrouter.ai/api/v1/chat/completions',
                 model="google/gemini-2.5-flash-preview",
                 system_prompt=None,
                 temperature=0.0,
                 max_tokens=1,
                 debug_mode=False,
                 api_request_delay=0.5,
                 max_api_retries=2):
        if api_key is None:
            load_dotenv()
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                logger.warning("No OpenRouter API key found. LLM strategy will fall back to random decisions.")
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.debug_mode = debug_mode
        self.api_request_delay = api_request_delay
        self.max_api_retries = max_api_retries
        
        # Configure logger level
        if debug_mode:
            logger.setLevel(logging.DEBUG)
        
        if system_prompt is None:
            self.system_prompt = (
                "You are playing a decision-based kingdom management game. Each turn, you are presented with a scenario (card) "
                "and must choose between two options: 'Left' or 'Right'. "
                "Each choice affects four metrics: Piety, Stability, Power, and Wealth. "
                "All metrics start at 50 and must always stay between 0 and 100. "
                "If any metric goes below 1 or above 99, the game ends immediately. "
                "Your objective is to survive as many turns as possible by making choices that keep all metrics within the safe range. "
                "Respond with only 'Left' or 'Right' to indicate your decision. No explanation is needed."
            )
        else:
            self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
    @property
    def name(self):
        return "openrouter"
    def make_decision(self, card, metrics, history):
        if not self.api_key:
            import random
            return random.choice(['Left', 'Right'])
        try:
            user_prompt = self._format_prompt(card, metrics, history)
            response = self._call_api(user_prompt)
            
            # Debug the response
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"OpenRouter API response: {json.dumps(response, indent=2)}")
                
            # Check if the response has the expected structure
            if not response:
                logger.warning("Empty API response")
                import random
                return random.choice(['Left', 'Right'])
                
            # Try to extract the completion from different possible response formats
            # OpenRouter might return different formats depending on the model
            try:
                # First check for error responses
                if 'error' in response:
                    error_msg = response.get('error', {}).get('message', 'Unknown error')
                    error_code = response.get('error', {}).get('code', 0)
                    
                    if error_code == 429:
                        # Rate limit error - handle gracefully
                        logger.warning(f"Rate limit exceeded: {error_msg}")
                        logger.info("Falling back to random decision due to rate limit")
                        import random
                        return random.choice(['Left', 'Right'])
                    else:
                        # Other API error
                        logger.warning(f"API error ({error_code}): {error_msg}")
                        raise KeyError(f"API error: {error_msg}")
                        
                # Process successful responses
                if 'choices' in response and response['choices']:
                    if 'message' in response['choices'][0]:
                        reply = response['choices'][0]['message']['content'].strip()
                    elif 'text' in response['choices'][0]:
                        reply = response['choices'][0]['text'].strip()
                    else:
                        logger.warning(f"Unknown choices format: {response['choices'][0]}")
                        raise KeyError("No message or text in choices")
                elif 'completion' in response:
                    reply = response['completion'].strip()
                elif 'output' in response:
                    reply = response['output'].strip()
                elif 'generated_text' in response:
                    reply = response['generated_text'].strip()
                else:
                    logger.warning(f"Unknown response format: {response}")
                    raise KeyError("No recognizable output format")
                    
                logger.debug(f"LLM reply: {reply}")
            except (KeyError, IndexError) as e:
                logger.warning(f"Error extracting reply from response: {str(e)}. Response: {response}")
                import random
                return random.choice(['Left', 'Right'])
                
            if reply.lower().startswith('left'):
                return 'Left'
            elif reply.lower().startswith('right'):
                return 'Right'
            else:
                logger.warning(f"Unexpected LLM response: {reply}. Falling back to random decision.")
                import random
                return random.choice(['Left', 'Right'])
        except Exception as e:
            logger.error(f"Error in LLM decision: {e}")
            import random
            return random.choice(['Left', 'Right'])
    def _format_prompt(self, card, metrics, history):
        try:
            character = get_card_value(card, 'Character')
            prompt = get_card_value(card, 'Prompt')
            left_choice = get_card_value(card, 'Left_Choice')
            left_piety = get_card_value(card, 'Left_Piety')
            left_stability = get_card_value(card, 'Left_Stability')
            left_power = get_card_value(card, 'Left_Power')
            left_wealth = get_card_value(card, 'Left_Wealth')
            right_choice = get_card_value(card, 'Right_Choice')
            right_piety = get_card_value(card, 'Right_Piety')
            right_stability = get_card_value(card, 'Right_Stability')
            right_power = get_card_value(card, 'Right_Power')
            right_wealth = get_card_value(card, 'Right_Wealth')
        except KeyError as e:
            logger.error(f"Error extracting card data: {e}")
            raise
        return (
            f"You are playing a decision-based strategy game. Here is the current card:\n"
            f"Character: {character}\n"
            f"Prompt: {prompt}\n"
            f"Left Choice: {left_choice} (Piety: {left_piety}, Stability: {left_stability}, "
            f"Power: {left_power}, Wealth: {left_wealth})\n"
            f"Right Choice: {right_choice} (Piety: {right_piety}, Stability: {right_stability}, "
            f"Power: {right_power}, Wealth: {right_wealth})\n\n"
            f"Current metrics:\n"
            f"Piety: {metrics['Piety']}, Stability: {metrics['Stability']}, "
            f"Power: {metrics['Power']}, Wealth: {metrics['Wealth']}\n\n"
            f"Respond with only 'Left' or 'Right' to indicate your decision. No explanation needed.\n"
        )
    def _call_api(self, user_prompt):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/torinvandenbulk/dynastai', # Add referrer to help with API tracking
            'User-Agent': 'DynastAI-Simulation/1.0.0'  # Add a custom user agent
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        logger.info(f"Calling OpenRouter API with model: {self.model}")
        
        # Get retry parameters from instance
        max_retries = self.max_api_retries
        retry_delay = self.api_request_delay
        
        @weave.op()
        def openrouter_call(api_url, headers, data, retries_left=max_retries):
            try:
                # Make API request
                response = requests.post(api_url, headers=headers, data=json.dumps(data))
                
                # Check for rate limit errors (HTTP 429)
                if response.status_code == 429:
                    reset_time = response.headers.get('X-RateLimit-Reset')
                    remaining = response.headers.get('X-RateLimit-Remaining', '0')
                    
                    # Log detailed rate limit information
                    logger.warning(f"Rate limit exceeded: {remaining} requests remaining, reset at {reset_time}")
                    
                    # If we still have retries left, sleep and retry
                    if retries_left > 0 and retry_delay > 0:
                        logger.info(f"Retrying after {retry_delay} seconds ({retries_left} retries left)")
                        import time
                        time.sleep(retry_delay)
                        return openrouter_call(api_url, headers, data, retries_left - 1)
                    else:
                        # Parse and return the error response
                        try:
                            return response.json()
                        except:
                            return {
                                "error": {
                                    "message": f"Rate limit exceeded, status code: {response.status_code}",
                                    "code": 429
                                }
                            }
                
                # Handle other errors
                elif response.status_code != 200:
                    logger.error(f"OpenRouter API error: {response.status_code} {response.text}")
                    
                    # Try to parse the error response
                    try:
                        return response.json()
                    except:
                        return {
                            "error": {
                                "message": f"API error: {response.status_code} - {response.text}",
                                "code": response.status_code
                            }
                        }
                
                # Process successful response
                try:
                    json_response = response.json()
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Raw API response: {json.dumps(json_response, indent=2)}")
                    return json_response
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.error(f"Raw response: {response.text}")
                    return {
                        "error": {
                            "message": f"JSON parse error: {str(e)}",
                            "code": 500
                        }
                    }
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                return {
                    "error": {
                        "message": f"Request error: {str(e)}",
                        "code": 500
                    }
                }
        
        try:
            return openrouter_call(self.api_url, headers, data)
        except Exception as e:
            logger.error(f"API call failed: {e}")
            # Return a minimal valid response format that will fail gracefully
            return {
                "error": {
                    "message": f"General error: {str(e)}",
                    "code": 500
                }
            } 