'''Description:
In this task, we implement a sentiment analysis model using text data. The dataset consists of product reviews that are classified into three sentiment categories: Positive, Negative, and Neutral. We preprocess the text data by cleaning, tokenizing, and removing special characters. Sentiment analysis is performed using the TextBlob library, and the results are visualized using bar charts to show the distribution of sentiments.

Steps Involved:
Data Collection: Collect a sample dataset of product reviews or text data.
Preprocessing: Clean the text data by removing URLs, special characters, and converting text to lowercase.
Sentiment Analysis: Perform sentiment analysis using TextBlob to classify the sentiment into Positive, Negative, or Neutral.
Data Visualization: Visualize the sentiment distribution using bar charts with Matplotlib and Seaborn. '''

# Import required libraries
import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Collect Sample Dataset
# Create a sample dataset of text reviews (you can replace this with your own dataset)
data = {
    'Review': [
        "I love this product! It's amazing and works perfectly.",
        "Worst experience ever. I will never buy this again.",
        "The quality is okay, but it could be better.",
        "Absolutely fantastic! Highly recommend it.",
        "Terrible. It broke after one use.",
        "The product is average, nothing special.",
        "I am so happy with this purchase. Great value for money.",
        "Disappointed. The description was misleading.",
        "It's just okay. Not worth the hype.",
        "Excellent! Exceeded my expectations."
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Preprocess Text Data
def clean_text(text):
    """Function to clean the text by removing special characters, URLs, and extra spaces."""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z ]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = text.strip()  # Remove leading/trailing spaces
    return text

# Apply cleaning function to the dataset
df['Cleaned_Review'] = df['Review'].apply(clean_text)

# Step 3: Perform Sentiment Analysis
def analyze_sentiment(text):
    """Function to analyze sentiment using TextBlob."""
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis function to the cleaned text
df['Sentiment'] = df['Cleaned_Review'].apply(analyze_sentiment)

# Step 4: Visualize Sentiment Distribution
# Count the number of each sentiment
sentiment_counts = df['Sentiment'].value_counts()

# Plot sentiment distribution
plt.figure(figsize=(5, 5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
plt.title("Sentiment Analysis Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Step 5: Display Final Results
print("Sample of the processed data:\n")
print(df)

# Optional: Save results to a CSV file
df.to_csv('sentiment_analysis_results.csv', index=False)
print("\nResults saved to 'sentiment_analysis_results.csv'")
