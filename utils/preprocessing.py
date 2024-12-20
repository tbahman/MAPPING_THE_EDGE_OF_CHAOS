def clean_gutenberg_text(text):
    start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    return text[start_idx + len(start_marker) : end_idx].strip() if start_idx != -1 and end_idx != -1 else text

def build_vocab(text):
    unique_chars = sorted(set(text))
    char2idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx2char = {idx: char for idx, char in enumerate(unique_chars)}
    return char2idx, idx2char

def tokenize(text, char2idx):
    return [char2idx[char] for char in text]

def create_dataset(tokens, seq_length):
    inputs, targets = [], []
    for i in range(len(tokens) - seq_length):
        inputs.append(tokens[i : i + seq_length])
        targets.append(tokens[i + 1 : i + seq_length + 1])
    return np.array(inputs), np.array(targets)
