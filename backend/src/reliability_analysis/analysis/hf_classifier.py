"""
Semantic Classifier using Hugging Face Zero-Shot pipelines.
Supports dynamically loading multiple models on-demand and caching them in memory.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class SemanticModelManager:
    """Singleton manager to cache loaded Hugging Face pipelines."""

    _pipelines = {}

    @classmethod
    def get_pipeline(cls, model_name: str):
        if model_name not in cls._pipelines:
            try:
                from transformers import pipeline

                logger.info(
                    f"Downloading/Loading Hugging Face model: {model_name}. This may take a moment..."
                )
                # Initialize pipeline for zero-shot classification
                pipe = pipeline("zero-shot-classification", model=model_name)
                cls._pipelines[model_name] = pipe
                logger.info(f"Successfully loaded {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                raise
        return cls._pipelines[model_name]

    @classmethod
    def batch_predict(
        cls, texts: List[str], model_name: str, candidate_labels: List[str]
    ) -> List[str]:
        """
        Runs zero-shot classification on a batch of texts and returns the top label for each.
        """
        if not texts:
            return []

        pipe = cls.get_pipeline(model_name)

        # Run classification
        results = pipe(texts, candidate_labels)

        # Pipeline output could be a single dict (if 1 text) or list of dicts
        if isinstance(results, dict):
            results = [results]

        predicted_labels = []
        for res in results:
            # 'labels' array is sorted by score descending, so [0] is the top prediction
            top_label = res["labels"][0]
            predicted_labels.append(top_label)

        return predicted_labels
