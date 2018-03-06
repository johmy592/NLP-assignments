import gensim
import numpy as np

def main():
    v1 = 'word2vec/oanc_vectors.bin'
    #v2 = 'word2vec/enwik9_vectors.bin'
    for model_name, vector_fn in [('OANC', v1)]:
        with open('data/toefl.txt') as f:
            model = gensim.models.KeyedVectors.load_word2vec_format(vector_fn, binary=True)
            preds, ground_truths = [], []
            for line in f:
                
                elements = line.split()
                word = elements[0]
                ground_truth = elements[1]
                synonyms = elements[2:]
                scores = []
                for synonym in synonyms:
                    try:
                        score = model.similarity(word, synonym)
                    except:
                        score = 0
                    scores.append(score)
                preds.append(np.argmax(scores))
                ground_truths.append(int(ground_truth))
            accuracy = np.sum(np.array(preds) == np.array(ground_truths)) / len(preds)
            print("%s accuracy: %.2f" % (model_name, accuracy))

if __name__ == "__main__":
    main()
