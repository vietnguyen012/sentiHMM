/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package SentiAnalysis;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

/**
 *
 * @author Admin
 */
public class SentimentPos {
    
    static String punctuations = ".,:;''``";
    static String[] non_suffix = {"action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism",
        "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"};
    static String[] verb_suffix = {"ate", "ify", "ise", "ize"};
    static String[] adj_suffix = {"able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"};
    static String[] adv_suffix = {"ward", "wards", "wise"};

    public static ArrayList<String> preprocess(Hashtable<Integer, String> vocab, String filepath) {
        //preprocess train data
        ArrayList<String> prep = new ArrayList<>();
        BufferedReader reader;
        try {
            reader = new BufferedReader(new FileReader(filepath));
            String word = reader.readLine();
            while (word != null) {
                if (word.split("\\s+").length == 0 || word.trim().isEmpty()) {
                    word = "--n--";
                    prep.add(word);
                    word = reader.readLine();
                    continue;
                } else if (vocab.containsValue(word.trim()) == false) {
                    word = assign_unk(word.trim());
                    prep.add(word.trim());
                    word = reader.readLine();
                    continue;
                } else {
                    prep.add(word.trim());
                }
                // read next line
                word = reader.readLine();
            }
            reader.close();
        } catch (IOException e) {
        }
        return prep;
    }

    public static ArrayList<String> preprocess_text(Hashtable<Integer, String> vocab, String[] text) {
        //process text add them to corpus for sentiment statement
        ArrayList<String> prep = new ArrayList<>();
        for (String word : text) {
            if (word.split("\\s+").length == 0 || word.trim().isEmpty()) {
                word = "--n--";
                prep.add(word);
            } else if (vocab.containsValue(word.trim()) == false) {
                word = assign_unk(word.trim());
                prep.add(word.trim());
            } else {
                prep.add(word.trim());
            }
        }
        return prep;
    }

    public static String assign_unk(String word) {
        //assign token "--unk word type--" to the ones not in vocab
        String[] list_letters = word.split("");

        for (int i = 0; i < list_letters.length; i++) {
            if (Character.isDigit(word.charAt(i))) {
                return "--unk_digit--";
            }
            if (punctuations.contains(list_letters[i])) {
                return "--unk_punct--";
            }
            if (Character.isUpperCase(word.charAt(i))) {
                return "--unk_upper--";
            }
        }
        for (String non_suffix1 : non_suffix) {
            if (word.endsWith(non_suffix1)) {
                return "--unk_noun--";
            }
        }
        for (String verb_suffix1 : verb_suffix) {
            if (word.endsWith(verb_suffix1)) {
                return "--unk_verb--";
            }
        }
        for (int i = 0; i < adv_suffix.length; i++) {
            if (word.endsWith(non_suffix[i])) {
                return "--unk_adv--";
            }
        }
        return "--unk--";
    }

    public static ArrayList<String> get_word_tag(String line, Hashtable<Integer, String> vocab) {
        //get the word,tag pair from line in file data
        String str[] = line.split("\t");
        ArrayList<String> output = new ArrayList<>();

        if (line.trim().isEmpty()) {
            String word = "--n--";
            String tag = "--s--";
            output.add(word);
            output.add(tag);
            return output;
        } else {
            String word = str[0].trim();
            String tag = str[1].trim();
            if (!vocab.containsValue(word)) {
                word = assign_unk(word);
            }
            output.add(word);
            output.add(tag);
            return output;
        }
    }

    public static ArrayList<String> read_data(String filepath) {
        //read line by line and add them to corpus
        ArrayList<String> corpus = new ArrayList<>();
        BufferedReader reader;
        try {
            reader = new BufferedReader(new FileReader(filepath));
            String line = reader.readLine();
            while (line != null) {

                corpus.add(line.trim());

                // read next line
                line = reader.readLine();
            }
            reader.close();
        } catch (IOException e) {
        }

        return corpus;
    }

    public static ArrayList<String> read_vocab(String filepath) {
        //read line by line and add words to vocab
        ArrayList<String> vocab = new ArrayList<>();
        BufferedReader reader;
        try {
            reader = new BufferedReader(new FileReader(filepath));
            String line = reader.readLine();
            while (line != null) {
                if (line.trim().isEmpty() == false) {
                    line = line.split("\n")[0].trim();
                    vocab.add(line);
                }
                // read next line
                line = reader.readLine();
            }
            reader.close();
        } catch (IOException e) {
        }

        return vocab;
    }

    public static int find_max_prob(double[][] best_probs, int j) {
        // find maximum prob in a given column of best_prob_matrix 
        double max_value = Double.NEGATIVE_INFINITY;
        int max_index = -1000;
        for (int i = 0; i < best_probs.length; i++) {
            if (best_probs[i][j] > max_value) {

                max_value = best_probs[i][j];
                max_index = i;
            }
        }
        //return index of max_prob
        return max_index;
    }

    public static ArrayList get_emssion_cnt_tag_cnt_trans_cnt(ArrayList<String> trainning_corpus, Hashtable<Integer, String> vocab) {
        //create 3 matrix
        //emission_counts include tag,word pairs and its freq
        //transition_counts include adjacent word pairs and its freq
        //tag_counts include adjacent tag pairs and its freq
        Hashtable< List<String>, Integer> emission_counts
                = new Hashtable<>();
        Hashtable< List<String>, Integer> transition_counts
                = new Hashtable<>();
        Hashtable< List<String>, Integer> tag_counts
                = new Hashtable<>();
        ArrayList<Object> output = new ArrayList<>();

        String prev_tag = "--s--";
        int i = 0;

        for (String word_tag : trainning_corpus) {
            i += 1;
            if (i % 50000 == 0) {
                System.out.println("word count=" + i);
            }
            ArrayList<String> w_t = get_word_tag(word_tag, vocab);
            String word = w_t.get(0);
            String tag = w_t.get(1);

            List<String> prev_tag_tag = Arrays.asList(prev_tag, tag);
            List<String> tag_word = Arrays.asList(tag, word);
            List<String> tg = Arrays.asList(tag);
            Integer oldcount = transition_counts.get(prev_tag_tag);
            if (oldcount == null) {

                transition_counts.put(prev_tag_tag, 1);
            } else {
                transition_counts.put(prev_tag_tag, oldcount + 1);
            }

            Integer oldcount2 = tag_counts.get(tg);
            if (oldcount2 == null) {

                tag_counts.put(tg, 1);
            } else {
                tag_counts.put(tg, oldcount2 + 1);
            }

            Integer oldcount1 = emission_counts.get(tag_word);
            if (oldcount1 == null) {
                emission_counts.put(tag_word, 1);
            } else {

                emission_counts.put(tag_word, oldcount1 + 1);
            }

            prev_tag = tag;

        }
        output.add(emission_counts);
        output.add(transition_counts);
        output.add(tag_counts);
        return output;
    }

    public static ArrayList<String> get_all_tag(Hashtable<List<String>, Integer> tag_counts) {
        //get all tag from tag_counts
        ArrayList<String> all_tags = new ArrayList<>();
        tag_counts.entrySet().stream().map((entry) -> entry.getKey()).forEachOrdered((key) -> {
            all_tags.add(key.get(0));
        });
        Collections.sort(all_tags);
        return all_tags;
    }

    public static double[][] create_transition_matrix(double alpha, int num_tags, ArrayList<String> all_tags,
            Hashtable<List<String>, Integer> tag_counts, Hashtable<List<String>, Integer> transition_counts) {
        //calculate probability of transition matrix by the formulas in slide
        double[][] A = new double[num_tags][num_tags];
        for (int i = 0; i < num_tags; i++) {
            for (int j = 0; j < num_tags; j++) {
                int count = 0;
                List<String> key = Arrays.asList(all_tags.get(i), all_tags.get(j));
                if (transition_counts != null) {
                    try {
                        count = transition_counts.get(key);
                    } catch (NullPointerException e) {
                        count = 0;
                    }
                }
                int count_prev_tag;
                try {
                    count_prev_tag = tag_counts.get(Arrays.asList(all_tags.get(i)));
                } catch (NullPointerException e) {
                    count_prev_tag = 0;
                }

                A[i][j] = (double) (count + alpha) / (count_prev_tag + alpha * num_tags);
            }
        }
        return A;
    }

    public static double[][] create_emission_matrix(double alpha, int num_tags, int num_words,
            ArrayList<String> all_tags, Hashtable<Integer, String> vocab,
            Hashtable< List<String>, Integer> emission_counts, Hashtable<List<String>, Integer> tag_counts) {
        //calculate emission matrix probabily by formulas in slide
        double[][] B = new double[num_tags][num_words];
        for (int i = 0; i < num_tags; i++) {
            for (int j = 0; j < num_words; j++) {
                int count;
                List<String> key = Arrays.asList(all_tags.get(i), vocab.get(j));
                try {
                    count = emission_counts.get(key);
                } catch (Exception e) {
                    count = 0;
                }
                int count_tag;
                try {
                    count_tag = tag_counts.get(Arrays.asList(all_tags.get(i)));
                } catch (Exception e) {
                    count_tag = 0;
                }
                B[i][j] = (count + alpha) / (count_tag + alpha * num_words);
            }
        }
        return B;
    }

    public static double[][] initialize(int num_tags, double neg_inf, double[][] B,
            double[][] A, int s_idx, Hashtable<String, Integer> inverse_vocab, ArrayList<String> prep) {
        //intialize best prob matrix and fill up the first column 
        double best_probs[][] = new double[num_tags][prep.size()];
        for (int i = 0; i < num_tags; i++) {
            if (A[s_idx][i] == 0) {
                best_probs[i][0] = neg_inf;
            } else {
                best_probs[i][0] = Math.log(A[s_idx][i]) + Math.log(B[i][inverse_vocab.get(prep.get(0))]);
            }
        }
        return best_probs;
    }

    public static ArrayList viterbi_forward(int num_tags, double[][] A, double[][] B, double neg_inf, ArrayList<String> prep,
            double best_probs[][], int best_paths[][], Hashtable<String, Integer> inverse_vocab) {
        ArrayList<Object> output = new ArrayList<>();
        for (int j = 1; j < prep.size(); ++j) {
            if (j % 5000 == 0) {
                System.out.println("words processed:" + j);
            }
            for (int i = 0; i < num_tags; i++) {
                double best_prob_j = neg_inf;
                int best_path_j = -1000000;
                for (int k = 0; k < num_tags; k++) {
                    //using the formulas calculating best prob for the remain columns
                    //use sum of log instead of product for numerical stability
                    double prob = best_probs[k][j - 1] + Math.log(A[k][i])
                            + Math.log(B[i][inverse_vocab.get(prep.get(j))]);
                    if (prob > best_prob_j) {
                        best_prob_j = prob;
                        best_path_j = k;
                    }
                }
                best_probs[i][j] = best_prob_j;
                best_paths[i][j] = best_path_j;

            }
        }
        output.add(best_probs);
        output.add(best_paths);
        return output;
    }

    public static String[] viterbi_backward(int best_paths[][], double neg_inf, int num_tags,
            double best_probs[][], ArrayList<String> states, ArrayList<String> prep) {
        int m = best_paths[0].length;
        int z[] = new int[m];
        double best_prob_for_last_word = neg_inf;
        String pred[] = new String[m];
        int last_index_k = -1;
        for (int k = 0; k < num_tags; k++) {
            if (best_probs[k][best_probs[k].length - 1] > best_prob_for_last_word) {
                best_prob_for_last_word = best_probs[k][best_probs[k].length - 1];
                z[m - 1] = k;//store maximum index of the last column 
            }
        }
        //backward index
        pred[m - 1] = states.get(z[m-1]);
        
        for (int i = prep.size() - 1; i >= 0; i--) {
            
            if ((i - 1) != -1) {
                //get best probability in column i to point to the index of best path in best path matrix 
                int pos_tag_for_word_i = best_paths[find_max_prob(best_probs, i)][i];
                z[i - 1] = best_paths[pos_tag_for_word_i][i];
                pred[i - 1] = states.get(pos_tag_for_word_i);//store pos
            }
        }
        return pred;
    }

    public static void predict(String[] pred, ArrayList<String> test_corpus) {
        //get predict and compare to ground truth to calculate accuracy
        double num_correct = 0;
        double total = 0;
        for (int i = 0; i < pred.length; i++) {
            String[] word_n_tag = test_corpus.get(i).split("\\s+");
            if (word_n_tag.length != 2) {
                continue;
            }
            String word = word_n_tag[0];
            String tag = word_n_tag[1];
            if (pred[i].trim().equals(tag.trim())) {
                num_correct += 1;
            }
            total += 1;
        }
        System.out.println("Accuracy:" + num_correct / total);
    }

    public static String[] load_predict(ArrayList<String> states, ArrayList<String> prep,
            Hashtable<Integer, String> vocab, int num_tags, double alpha, double[][] A, double[][] B, Hashtable<String, Integer> inverse_vocab) {

        int s_idx = states.indexOf("--s--");
        double best_probs[][] = initialize(num_tags, alpha, B, A, s_idx, inverse_vocab, prep);
        int best_paths[][] = new int[num_tags][prep.size()];

        double neg_inf = Double.NEGATIVE_INFINITY;

        //viterbi forward
        ArrayList<Object> output = viterbi_forward(num_tags, A, B, neg_inf, prep, best_probs, best_paths, inverse_vocab);
        best_probs = (double[][]) output.get(0);
        best_paths = (int[][]) output.get(1);

        //viterbi backwawrd
        String[] pred = viterbi_backward(best_paths, neg_inf, num_tags, best_probs, states, prep);
        return pred;
    }

    public static boolean is_adjective(String tag) {
        return "JJ".equals(tag) | "JJR".equals(tag) | "JJS".equals(tag);
    }

    public static boolean is_adverb(String tag) {
        return "RB".equals(tag) | "RBR".equals(tag) | "RBS".equals(tag);
    }

    public static boolean is_noun(String tag) {
        return "NN".equals(tag) | "NNS".equals(tag) | "NNP".equals(tag) | "NNPS".equals(tag);
    }

    public static boolean is_verb(String tag) {
        return "VB".equals(tag) | "VBD".equals(tag) | "VBG".equals(tag) | "VBN".equals(tag) | "VBP".equals(tag) | "VBZ".equals(tag);
    }

    public static boolean is_valid(String tag) {
        //check word is noun, adverb,verb or adjective
        return is_noun(tag) | is_adverb(tag) | is_verb(tag) | is_adjective(tag);
    }

    public static ArrayList<String> filter(ArrayList<String> words, String[] pred_tag) {
        //get all words that are valid
        ArrayList<String> filter_words = new ArrayList<>();
        int count = 0;
        for (String tag : pred_tag) {
            if (is_valid(tag)) {
                filter_words.add(words.get(count));
            }
            count += 1;
        }
        return filter_words;
    }

    public static double[] get_score(ArrayList<String> filter_words, SentiWordNet SWN) {
        //extract score from sentiwordnet
        double[] score = new double[filter_words.size()];
        for (int i = 0; i < filter_words.size(); i++) {
            score[i] = SWN.extract(filter_words.get(i));
        }
        return score;
    }

    public static double get_neg_score(double[] score) {
        //sum negative score
        double total = 0;
        for (double d : score) {
            if (d < 0) {
                total += d;
            }
        }
        return total;
    }

    public static String get_tweet_sentment_from_score(double posScore, double negScore) {
        // compare positive score and negative score to get result
        double epsilon = 0.23;//a margin for neutral statement
        if (posScore - negScore > epsilon) {
            return "positive";
        } else if (- epsilon <= posScore - negScore && posScore - negScore <= epsilon) {
            return "neutral";
        } else {
            return "negative";
        }
    }

    public static double get_pos_score(double[] score) {
        //sum pos score
        double total = 0;
        for (double d : score) {
            if (d > 0) {
                total += d;
            }
        }
        return total;
    }

    public static void main(String[] args) throws IOException {
        // TODO code application logic here

        SentiWordNet SWN = new SentiWordNet();
        ArrayList<String> trainning_corpus;
        ArrayList<String> test_corpus;
        ArrayList<String> voc_l;
        trainning_corpus = read_data("WSJ_02-21.pos");
        test_corpus = read_data("WSJ_24.pos");
        voc_l = read_vocab("hmm_vocab.txt");
        Hashtable<Integer, String> vocab = new Hashtable<>();
        Hashtable<String, Integer> inverse_vocab = new Hashtable<>();

        //get vocab and inverse vocab dictionary
        for (int i = 0; i < voc_l.size(); i++) {
            vocab.put(i, voc_l.get(i));
            inverse_vocab.put(voc_l.get(i), i);
        }
        //create count matrixes
        Hashtable< List<String>, Integer> emission_counts;
        Hashtable< List<String>, Integer> transition_counts;
        Hashtable< List<String>, Integer> tag_counts;

        ArrayList<Object> output = get_emssion_cnt_tag_cnt_trans_cnt(trainning_corpus, vocab);
        emission_counts = (Hashtable<List<String>, Integer>) output.get(0);
        transition_counts = (Hashtable<List<String>, Integer>) output.get(1);
        tag_counts = (Hashtable<List<String>, Integer>) output.get(2);

        ArrayList<String> all_tags = get_all_tag(tag_counts);
        ArrayList<String> states = all_tags;

        int num_tags = all_tags.size();
        double alpha = (double) 0.001;
        double[][] A = create_transition_matrix(alpha, num_tags, all_tags, tag_counts, transition_counts);

        int num_words = vocab.size();
        double[][] B = create_emission_matrix(alpha, num_tags, num_words, all_tags, vocab, emission_counts, tag_counts);
        ArrayList<String> prep = preprocess(vocab, "test.words");

        String[] pred = load_predict(states, prep, vocab, num_tags, alpha, A, B, inverse_vocab);
        predict(pred, test_corpus);

        for (int j = 0; j < 100; j++) {
            System.out.println(prep.get(j) + " " + pred[j]);
        }
        Scanner sc = new Scanner(System.in);
        
        //Sentiment analysis section
        while (true) {
            
            System.out.println("Enter comment:");
            String comment = sc.nextLine();
            String[] ws = comment.split("\\s+");
        
            ArrayList<String> words = preprocess_text(vocab, ws);
            
            System.out.println("predicted tags:");
            String[] pred_tag = load_predict(states, words, vocab, num_tags, alpha, A, B, inverse_vocab);
            for (String pr : pred_tag) {
                System.out.print(pr+" ");
            }
            System.out.println("\nfiltered words:");
            ArrayList<String> filter_words = filter(words, pred_tag);
            filter_words.forEach((fil) -> {
                System.out.print(fil+" ");
            });
            double[] score = get_score(filter_words, SWN);
            System.out.println("\nscore each filltered word:");
            for (int j = 0; j < score.length; j++) {
                System.out.print(score[j]+" ");
            }
            
            double neg = Math.abs(get_neg_score(score));
            double pos = get_pos_score(score);
            String res = get_tweet_sentment_from_score(pos, neg);
            
            boolean in_valid = true;
            String choice = null;
            System.out.println("\nsentiment statement:"+res);
            while(in_valid){
                System.out.println("continue?(Y/N)");
                choice = sc.nextLine();
                if(choice.endsWith("Y")|choice.endsWith("N")){
                    in_valid = false;
                }
            }
             if(!"N".equals(choice)){
                } else {
                 break;
            }
        }
    }
}
