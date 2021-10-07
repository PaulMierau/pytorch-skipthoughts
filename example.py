from skipthoughts import Encoder

dirStr = 'models'
encoder = Encoder(dirStr)

sentences = ["Hey, how are you?", "This sentence is a lie"]

encodedSentences = encoder.encode(sentences)
print(encodedSentences)