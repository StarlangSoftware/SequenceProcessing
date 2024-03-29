package SequenceProcessing.Sequence;

import Corpus.Corpus;
import Corpus.Sentence;
import Dictionary.VectorizedWord;
import Util.FileUtils;
import Math.Vector;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class SequenceCorpus extends Corpus {

    /**
     * Constructor which takes a file name {@link String} as an input and reads the file line by line. It takes each word of the line,
     * and creates a new {@link VectorizedWord} with current word and its label. It also creates a new {@link Sentence}
     * when a new sentence starts, and adds each word to this sentence till the end of that sentence.
     *
     * @param fileName File which will be read and parsed.
     */
    public SequenceCorpus(String fileName) {
        super();
        String line, word;
        VectorizedWord newWord;
        Sentence newSentence = null;
        try {
            InputStreamReader fr = new InputStreamReader(FileUtils.getInputStream(fileName));
            BufferedReader br = new BufferedReader(fr);
            line = br.readLine();
            while (line != null) {
                String[] items = line.split(" ");
                word = items[0];
                if (word.equals("<S>")) {
                    if (items.length == 2){
                        newSentence = new LabelledSentence(items[1]);
                    } else {
                        newSentence = new Sentence();
                    }
                } else {
                    if (word.equals("</S>")) {
                        addSentence(newSentence);
                    } else {
                        if (items.length == 2) {
                            newWord = new LabelledVectorizedWord(word, items[1]);
                        } else {
                            newWord = new VectorizedWord(word, new Vector(300,0));
                        }
                        if (newSentence != null){
                            newSentence.addWord(newWord);
                        }
                    }
                }
                line = br.readLine();
            }
        } catch (IOException ignored) {
        }
    }

    public ArrayList<String> getClassLabels() {
        boolean sentenceLabelled = false;
        ArrayList<String> classLabels = new ArrayList<>();
        if (sentences.get(0) instanceof LabelledSentence){
            sentenceLabelled = true;
        }
        for (int i = 0; i < sentenceCount(); i++) {
            if (sentenceLabelled){
                LabelledSentence sentence = (LabelledSentence) sentences.get(i);
                if (!classLabels.contains(sentence.getClassLabel())) {
                    classLabels.add(sentence.getClassLabel());
                }
            } else {
                Sentence sentence = sentences.get(i);
                for (int j = 0; j < sentence.wordCount(); j++){
                    LabelledVectorizedWord word = (LabelledVectorizedWord) sentence.getWord(j);
                    if (!classLabels.contains(word.getClassLabel())) {
                        classLabels.add(word.getClassLabel());
                    }
                }
            }
        }
        return classLabels;
    }

}
