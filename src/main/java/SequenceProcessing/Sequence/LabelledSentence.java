package SequenceProcessing.Sequence;

import Corpus.Sentence;

public class LabelledSentence extends Sentence {

    private final String classLabel;

    public LabelledSentence(String classLabel){
        super();
        this.classLabel = classLabel;
    }

    public String getClassLabel(){
        return classLabel;
    }
}
