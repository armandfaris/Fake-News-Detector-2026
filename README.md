Building this was more than just training a model, it was about ensuring Data Integrity through a cleaning process. To transform raw data to an accurate prediction, I implemented a detailed Data Preparation and Modelling pipeline:

1) Noise Reduction and Data Preparation: Applied Tokenization, Stop-word removal, and Lemmatization to isolate core meanings.

2) Statistical Filtering: Used max_df = 0.5 to ignore high-frequency words that don't add value to the model. 3) Logarithmic Scaling: Applied log-scaling to TF-IDF to balance the importance of words across 100,000 articles. 4) Word2Vec (Semantic Layer): Learns the contextual neighbourhood of words that help the model to understand the relationship between terms rather than just their frequency.

5) Logistic Regression (Classifier): A high-speed, interpretable model specifically chosen for Binary Classification. It maps the combined feature vectors into a probability between 0 and 1 to distinguish between Real and Fake.

The Result? A two-layer verification app that uses Explainable Al which is LIME to tell you exactly why a piece of news is flagged.

Try the Live System here: https://Inkd.in/gfC_gEfB
