package SequenceProcessing.Sequence;

import Math.Vector;

public class EmbeddedWordInstance extends EmbeddedWord{

    private String classLabel;

    public EmbeddedWordInstance(String word, Vector embedding, String classLabel) {
        super(word, embedding);
        this.classLabel = classLabel;
    }

    public EmbeddedWordInstance(String word, String classLabel) {
        super(word);
        this.classLabel = classLabel;
    }

    public String getClassLabel(){
        return classLabel;
    }
}
