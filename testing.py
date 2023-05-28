import matplotlib.pyplot as plt

mapping = {
    "fear": "negative",
    "disgust": "negative",
    "happiness": "positive",
    "anger": "negative",
    "sadness": "negative",
}

# Count the number of occurrences for each category
categories = set(mapping.values())
category_counts = {category: list(mapping.values()).count(category) for category in categories}

# Plotting the bar chart
fig, ax = plt.subplots()
emotion_labels = list(mapping.keys())
emotion_values = [category_counts[mapping[emotion]] for emotion in emotion_labels]
colors = ['r' if mapping[emotion] == 'negative' else 'g' for emotion in emotion_labels]

ax.bar(emotion_labels, emotion_values, color=colors)
ax.set_xlabel('Emotions')
ax.set_ylabel('Counts')
ax.set_title('Emotion Category Distribution')

# Adding a legend for the colors
negative_patch = plt.Line2D([0], [0], color='r', label='Negative')
positive_patch = plt.Line2D([0], [0], color='g', label='Positive')
ax.legend(handles=[negative_patch, positive_patch])

plt.show()

import matplotlib.pyplot as plt
from wordcloud import WordCloud

mapping = {
    "fear": "negative",
    "disgust": "negative",
    "happiness": "positive",
    "anger": "negative",
    "sadness": "negative",
}

# Generate a string of emotions based on their mapping
emotions = ' '.join(mapping.keys())

# Create the WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(emotions)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Emotion Mapping Word Cloud')
plt.show()
