package SequenceProcessing.Sequence;

import Dictionary.Word;
import Math.Vector;

public class EmbeddedWord extends Word {

    private Vector embedding;

    public EmbeddedWord(String word, Vector embedding) {
        super(word);
        this.embedding = embedding;
    }

    public EmbeddedWord(String word) {
        super(word);
    }

    public void setEmbedding(Vector embedding){
        this.embedding = embedding;
    }

    public Vector getEmbedding(){
        return embedding;
    }
}
