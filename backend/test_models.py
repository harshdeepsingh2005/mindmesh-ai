import asyncio
import os
import sys

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(__file__))

from app.services.text_embeddings import get_embedding_engine
from app.services.emotion_detection import get_emotion_engine
from app.services.topic_discovery import get_topic_engine
from app.services.student_clustering import get_cluster_engine
from app.services.anomaly_detection import get_anomaly_engine
from app.services.anomaly_detection import BehavioralFeatureVector

def test_models():
    print("=== Loading Retrained Models ===")
    embed_engine = get_embedding_engine()
    print(f"Text Embeddings: Fitted={embed_engine.is_fitted}, Vocab Size={embed_engine.vocabulary_size}")

    emotion_engine = get_emotion_engine()
    print(f"\n--- Emotion Clusters ({len(emotion_engine.clusters)} clusters) ---")
    for c in emotion_engine.clusters:
        print(f"  [{c.label}] -> Top terms: {', '.join(c.top_terms[:5])} (Size: {c.size})")

    topic_engine = get_topic_engine()
    print(f"\n--- Discovered Topics ({len(topic_engine.topics)} topics) ---")
    for t in topic_engine.topics:
        print(f"  [{t.label}] -> Top terms: {', '.join(t.top_terms[:5])}")

    cluster_engine = get_cluster_engine()
    print(f"\n--- Student Behavioral Clusters ---")
    config = cluster_engine.get_config()
    print(f"  Mode: {config['type']}, Fitted: {config['is_fitted']}, Number of Clusters: {config.get('n_clusters_actual', config.get('n_clusters'))}")

    anomaly_engine = get_anomaly_engine()
    config = anomaly_engine.get_config()
    print(f"\n--- Anomaly Detection ---")
    print(f"  Config: {config}")

    print("\n=== Testing Live Inference ===")
    test_texts = [
        "I'm feeling really stressed about exams and I can't sleep at all. Everything is terrible.",
        "Today was a great day! I hung out with my friends and we played video games.",
        "Nothing much happened today. Just went to class and did some homework."
    ]

    for text in test_texts:
        print(f"\nText: '{text}'")
        
        # 1. Emotion
        emotion_res = emotion_engine.predict(text)
        print(f"  -> Emotion: [{emotion_res.cluster_label}] (Confidence: {emotion_res.confidence_score:.2f})")
        
        # 2. Topic
        topic_res = topic_engine.predict(text)
        print(f"  -> Topic: [{topic_res.topic_label}] (Confidence: {topic_res.confidence:.2f})")

    print("\n=== Testing Anomaly Profiling ===")
    # Synthetic high-risk student profile
    high_risk_fv = BehavioralFeatureVector(
        student_id="test_high_risk",
        avg_sentiment=-0.9, 
        sentiment_std=0.8, 
        negative_ratio=0.9,
        dominant_cluster=1,
        emotion_entropy=0.5,
        distress_ratio=0.8,
        entries_per_week=0.5,
        journal_ratio=0.8,
        days_since_last_entry=14,
        avg_mood=1.2,
        mood_std=2.0,
        high_risk_flags=3
    )
    
    # Synthetic normal student profile
    normal_fv = BehavioralFeatureVector(
        student_id="test_normal",
        avg_sentiment=0.5, 
        sentiment_std=0.2, 
        negative_ratio=0.1,
        dominant_cluster=0,
        emotion_entropy=2.0,
        distress_ratio=0.05,
        entries_per_week=5.0,
        journal_ratio=0.3,
        days_since_last_entry=1,
        avg_mood=4.0,
        mood_std=0.5,
        high_risk_flags=0
    )

    norm_res = anomaly_engine.predict(normal_fv)
    print(f"  Normal Student Profile:")
    print(f"    Is Anomaly? {norm_res.is_anomaly}")
    print(f"    Risk Level: {norm_res.risk_level}")
    print(f"    Score: {norm_res.anomaly_score}")

    high_res = anomaly_engine.predict(high_risk_fv)
    print(f"  High Risk Student Profile:")
    print(f"    Is Anomaly? {high_res.is_anomaly}")
    print(f"    Risk Level: {high_res.risk_level}")
    print(f"    Score: {high_res.anomaly_score}")
    print(f"    Top Contributing Features: {high_res.contributing_features}")

if __name__ == "__main__":
    test_models()
