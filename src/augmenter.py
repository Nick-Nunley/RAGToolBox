import os
import json
from typing import List, Optional
from pathlib import Path
from src.retriever import Retriever


class Augmenter:
    """
    Augmenter class for generating responses using retrieved chunks and LLM.
    
    This class takes retrieved chunks from the Retriever and combines them
    with the user query to generate a coherent response using a language model.
    """
    
    def __init__(self, model_name: str = "google/gemma-2-2b-it", api_key: Optional[str] = None, use_local: bool = False):
        """
        Initialize the Augmenter.
        
        Args:
            model_name: Name of the model to use (default: google/gemma-2-2b-it)
            api_key: API key for Hugging Face (defaults to HUGGINGFACE_API_KEY env var)
            use_local: Whether to use local model (True) or Hugging Face API (False)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.use_local = use_local
        
        # Initialize model based on preference
        if use_local:
            self._initialize_local_model()
        else:
            if not self.api_key:
                print("Warning: No API key provided. Some models may not work without authentication.")
            self._initialize_api_client()
    
    def _initialize_api_client(self):
        """Initialize the Hugging Face inference client."""
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=self.api_key)
        except ImportError:
            raise ImportError("huggingface_hub package is required. Install with: pip install huggingface_hub")
        except Exception as e:
            raise RuntimeError(f"Error initializing Hugging Face client: {str(e)}")
    
    def _initialize_local_model(self):
        """Initialize the local model using transformers library."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("Model loaded successfully!")
            
        except ImportError:
            raise ImportError("Transformers package is required. Install with: pip install transformers torch")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
    
    def _format_prompt(self, query: str, retrieved_chunks: List[str]) -> str:
        """
        Format the query and retrieved chunks into a prompt for the LLM.
        
        Args:
            query: The user's original query
            retrieved_chunks: List of retrieved text chunks
            
        Returns:
            Formatted prompt string
        """
        # Create context from retrieved chunks
        context = "\n\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(retrieved_chunks)])
        
        # Format the prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context. 
Please incorporate ALL applicable context into your response. Feel free to combine context if you need to. Use only the information from the context to answer the question. If the context doesn't contain 
enough information to answer the question, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def _call_llm(self, prompt: str, temperature: float = 0.25, max_new_tokens: int = 200) -> str:
        """
        Call the language model with the formatted prompt.
        
        Args:
            prompt: The formatted prompt to send to the LLM
            temperature: Controls randomness in generation (0.0 = deterministic, 1.0 = very random)
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response from the LLM
        """
        if self.use_local:
            return self._call_local_model(prompt, temperature, max_new_tokens)
        else:
            return self._call_huggingface_api(prompt, temperature, max_new_tokens)
    
    def _call_local_model(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 200) -> str:
        """Call the local model using transformers."""
        try:
            import torch
            
            # Tokenize the prompt
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove the input prompt)
            generated_text = response[len(prompt):].strip()
            
            return generated_text if generated_text else "I don't have enough information to provide a detailed answer."
            
        except Exception as e:
            raise RuntimeError(f"Error calling local model: {str(e)}")
    
    def _call_huggingface_api(self, prompt: str, temperature: float = 0.25, max_new_tokens: int = 200) -> str:
        """Call Hugging Face inference API."""
        try:
            # Use the InferenceClient to generate text using chat completions
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_new_tokens,
                temperature=temperature
            )
            
            # Extract the response from the completion
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            # Provide helpful error message for common issues
            error_str = str(e).lower()
            if "404" in error_str or "not found" in error_str or "stopiteration" in error_str:
                raise RuntimeError(f"Model '{self.model_name}' is not available on Hugging Face's inference API. "
                                 f"Try using a different model like 'deepseek-ai/DeepSeek-V3-0324', 'meta-llama/Llama-2-7b-chat-hf', or set use_local=True to use local models.")
            elif "authentication" in error_str or "token" in error_str:
                raise RuntimeError(f"Authentication error: {str(e)}. Please check your HUGGINGFACE_API_KEY environment variable.")
            else:
                raise RuntimeError(f"Error calling Hugging Face API: {str(e)}")
    
    def generate_response(self, query: str, retrieved_chunks: List[str], temperature: float = 0.25, max_new_tokens: int = 200) -> str:
        """
        Generate a response using the retrieved chunks as context.
        
        Args:
            query: The user's original query
            retrieved_chunks: List of retrieved text chunks from the Retriever
            temperature: Controls randomness in generation (0.0 = deterministic, 1.0 = very random)
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response from the LLM
        """
        if not retrieved_chunks:
            return "I don't have enough information to answer your question. Please try rephrasing or expanding your query."
        
        # Format the prompt
        prompt = self._format_prompt(query, retrieved_chunks)
        
        # Call the LLM
        response = self._call_llm(prompt, temperature, max_new_tokens)
        
        return response
    
    def generate_response_with_sources(self, query: str, retrieved_chunks: List[str], temperature: float = 0.25, max_new_tokens: int = 200) -> dict:
        """
        Generate a response with source information.
        
        Args:
            query: The user's original query
            retrieved_chunks: List of retrieved text chunks from the Retriever
            temperature: Controls randomness in generation (0.0 = deterministic, 1.0 = very random)
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing response and source information
        """
        response = self.generate_response(query, retrieved_chunks, temperature, max_new_tokens)
        
        return {
            "response": response,
            "sources": retrieved_chunks,
            "num_sources": len(retrieved_chunks),
            "query": query,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens
        }

if __name__ == "__main__":

    ### Test code
    retriever = Retriever(
        embedding_model = 'fastembed',
        db_path = Path('assets/kb/embeddings/embeddings.db')
        )

    augmenter = Augmenter()
    
    test_prompt = 'Can you tell me about LIFU? What does it stand for and how does it work?'

    context = retriever.retrieve(test_prompt)

    response = augmenter.generate_response(
        test_prompt,
        context
        )
    
    print(response)
