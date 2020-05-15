class Utils:
    def __init__(self, de, en):
        self.german_model = de
        self.english_model = en
    def tokenize_german(self,text):
        """
        Tokenizes German text from a string into a list of strings (tokens) and reverses it
        """
        return [tok.text for tok in self.german_model.tokenizer(text)][::-1]

    def tokenize_english(self, text):
        """
        Tokenizes English text from a string into a list of strings (tokens)
        """
        return [tok.text for tok in self.english_model.tokenizer(text)]
