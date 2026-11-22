# importing pipeline function, which simplifies using pretrained AI models.
from transformers import pipeline

print("Loading AI model...")

# classifier -> This loads a pretrained DistilBERT model for sentimental analysis.
# device=-1 -> This specifies that the model should run on CPU. If you have a GPU, you can set device=0 to use it.
classifier = pipeline(task="sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",device=-1)

print("AI model loaded successfully!")

while True:
    text = input("Enter text to analyze sentiment (or type 'exit' to quit): ")
    if text.lower() == 'exit':
        print("Exiting the program.")
        break
    if not text.strip():
        print("Please enter some text for analysis.")
        continue
    results = classifier(text)[0]
    sentiment = results['label']
    confidence = results['score'] * 100
    print(f"Sentiment: {sentiment}, Confidence: {confidence:.2f}%")
    print("-"*30)