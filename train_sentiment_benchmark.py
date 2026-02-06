import torch
import torch.nn as nn
import torch.optim as optim
from sa_nn import export_sa_nn

def create_comprehensive_sentiment_data():
    # Expanded vocabulary for better sentiment analysis
    vocab = {
        # Positive words
        "good": 0, "great": 1, "excellent": 2, "amazing": 3, "fantastic": 4,
        "awesome": 5, "wonderful": 6, "perfect": 7, "love": 8, "best": 9,
        "beautiful": 10, "brilliant": 11, "outstanding": 12, "superb": 13, "marvelous": 14,
        "incredible": 15, "phenomenal": 16, "terrific": 17, "splendid": 18, "fabulous": 19,
        
        # Negative words
        "bad": 20, "terrible": 21, "awful": 22, "horrible": 23, "worst": 24,
        "hate": 25, "disgusting": 26, "annoying": 27, "boring": 28, "sad": 29,
        "disappointing": 30, "mediocre": 31, "poor": 32, "weak": 33, "dull": 34,
        "frustrating": 35, "unpleasant": 36, "regrettable": 37, "abysmal": 38, "pathetic": 39,
        
        # Neutral/common words
        "movie": 40, "film": 41, "story": 42, "plot": 43, "acting": 44,
        "performance": 45, "character": 46, "characters": 47, "scene": 48, "scenes": 49,
        "very": 50, "really": 51, "quite": 52, "extremely": 53, "so": 54,
        "not": 55, "but": 56, "and": 57, "the": 58, "this": 59
    }
    
    positive_reviews = [
        "this movie is great and excellent",
        "amazing film with wonderful story",
        "love the acting and characters",
        "fantastic plot and amazing performance",
        "excellent movie with great acting",
        "awesome film love the characters",
        "wonderful story and perfect performance",
        "amazing movie great film",
        "excellent acting and awesome characters",
        "fantastic movie love it",
        "brilliant performance and outstanding story",
        "beautiful cinematography and superb acting",
        "incredible movie with phenomenal effects",
        "marvelous story and terrific acting",
        "splendid film with fabulous characters"
    ]
    
    negative_reviews = [
        "this movie is terrible and awful",
        "horrible film with boring story",
        "hate the acting and characters",
        "worst plot and disgusting performance",
        "terrible movie with bad acting",
        "awful film hate the characters",
        "boring story and worst performance",
        "horrible movie bad film",
        "terrible acting and annoying characters",
        "worst movie hate it",
        "disappointing movie with mediocre plot",
        "poor acting and weak story",
        "dull film with unpleasant characters",
        "frustrating plot and regrettable acting",
        "abysmal movie with pathetic performance"
    ]
    
    def text_to_features(text, vocab):
        features = [0] * len(vocab)
        words = text.lower().split()
        for word in words:
            if word in vocab:
                features[vocab[word]] = 1
        return features
    
    # Prepare training data
    X_train = []
    y_train = []
    
    # Positive examples (label 1)
    for review in positive_reviews:
        X_train.append(text_to_features(review, vocab))
        y_train.append(1)  # Positive sentiment
    
    # Negative examples (label 0)
    for review in negative_reviews:
        X_train.append(text_to_features(review, vocab))
        y_train.append(0)  # Negative sentiment
    
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long), vocab

class EmbeddedSentimentModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, 20)  # Smaller first layer for embedded
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 10)  # Smaller second layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 2)   # Output: 0=negative, 1=positive
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

X_train, y_train, vocab = create_comprehensive_sentiment_data()
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}")
print(f"Training samples: {len(X_train)}")

model = EmbeddedSentimentModel(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training embedded sentiment model...")

for epoch in range(25):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/25], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    predictions = model(X_train)
    _, predicted = torch.max(predictions.data, 1)
    accuracy = (predicted == y_train).float().mean()
    print(f'Training Accuracy: {accuracy:.4f}')

# Export the model using SA-NN with PROGMEM support
print("\nExporting model to SA-NN format with PROGMEM...")
export_sa_nn(model, vocab=vocab, filename="sentiment_model.h", use_progmem=True)
print("Embedded sentiment model with PROGMEM exported successfully!")

# Test with some new examples to verify
test_reviews = [
    "this movie is great",
    "terrible film hate it",
    "amazing and wonderful story",
    "boring and awful",
    "good movie with nice acting",
    "bad plot and poor performance"
]

print("\nTesting trained model on new examples:")
model.eval()
with torch.no_grad():
    for review in test_reviews:
        # Convert to feature vector
        features = [0] * vocab_size
        words = review.lower().split()
        for word in words:
            if word in vocab:
                features[vocab[word]] = 1
        
        x_test = torch.tensor([features], dtype=torch.float32)
        output = model(x_test)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        
        sentiment = "POSITIVE" if prediction.item() == 1 else "NEGATIVE"
        confidence = probabilities[0][prediction.item()].item()
        
        print(f"Review: '{review}' -> {sentiment} (confidence: {confidence:.3f})")
